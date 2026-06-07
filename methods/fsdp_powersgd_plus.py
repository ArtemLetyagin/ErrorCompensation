import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPPowerSGDPlusEFState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 2
    block_cols: int = 4096
    min_compression_rate: float = 2.0

    # PowerSGD+ safeguard
    svd_refresh_period: int = 50
    start_compression_iter: int = 10

    # Standard error feedback
    use_error_feedback: bool = True

    eps: float = 1e-8

    residuals: dict = field(default_factory=dict)
    q_memory: dict = field(default_factory=dict)

    hook_call_id: int = 0
    iter_idx: int = 0

    summary: dict = field(default_factory=dict)
    svd_stats: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}
        self.svd_stats = {}
        self.hook_call_id = 0
        self.iter_idx += 1

    def get_residual(self, tensor: torch.Tensor):
        key = self.hook_call_id

        residual = self.residuals.get(key)

        if (
            residual is None
            or residual.shape != tensor.shape
            or residual.device != tensor.device
            or residual.dtype != tensor.dtype
        ):
            residual = torch.zeros_like(tensor)
            self.residuals[key] = residual

        return residual

    def get_q_memory(self, hook_id, cols, rank, device, dtype):
        q = self.q_memory.get(hook_id)

        if (
            q is None
            or q.shape != (cols, rank)
            or q.device != device
            or q.dtype != dtype
        ):
            q = torch.randn(cols, rank, device=device, dtype=dtype)
            q = orthogonalize(q)
            self.q_memory[hook_id] = q

        return q

    def save_q_memory(self, hook_id, q):
        self.q_memory[hook_id] = q.detach()

    def should_compress(self):
        return self.iter_idx >= self.start_compression_iter

    def should_svd_refresh(self):
        if self.svd_refresh_period <= 0:
            return False

        return (
            self.should_compress()
            and self.iter_idx > 0
            and self.iter_idx % self.svd_refresh_period == 0
        )

    def add_record(self, op_name, original_bytes, compressed_bytes, latency_ms):
        key = str(original_bytes)

        if op_name not in self.summary:
            self.summary[op_name] = {}

        if key not in self.summary[op_name]:
            self.summary[op_name][key] = {
                "count": 0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,

                "original_msg_size_bytes": original_bytes,
                "original_msg_size_str": format_bytes(original_bytes),

                "compressed_msg_size_bytes": compressed_bytes,
                "compressed_msg_size_str": format_bytes(compressed_bytes),

                "compression_ratio": (
                    original_bytes / compressed_bytes
                    if compressed_bytes > 0
                    else None
                ),

                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
            }

        rec = self.summary[op_name][key]
        rec["count"] += 1
        rec["total_latency_ms"] += latency_ms
        rec["avg_latency_ms"] = rec["total_latency_ms"] / rec["count"]
        rec["total_original_bytes"] += original_bytes
        rec["total_compressed_bytes"] += compressed_bytes

    def add_svd_record(self, hook_id, used_svd, rows, cols, rank):
        key = str(hook_id)

        if key not in self.svd_stats:
            self.svd_stats[key] = {
                "count": 0,
                "svd_refresh_count": 0,
                "rows": rows,
                "cols": cols,
                "rank": rank,
            }

        rec = self.svd_stats[key]
        rec["count"] += 1

        if used_svd:
            rec["svd_refresh_count"] += 1


def orthogonalize(mat: torch.Tensor):
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def compute_global_svd_q(
    matrix: torch.Tensor,
    rank: int,
    state: FSDPPowerSGDPlusEFState,
):
    """
    SVD safeguard for PowerSGD+.

    Instead of computing full SVD of the full flattened gradient matrix directly,
    we compute the right singular subspace from the global Gram matrix:

        G = sum_i M_i^T M_i

    Then the top eigenvectors of G correspond to the top right singular vectors
    of the distributed matrix stack.

    This gives a shared Q on all ranks.
    """
    dtype = matrix.dtype
    cols = matrix.shape[1]

    gram = matrix.float().t().matmul(matrix.float())

    dist.all_reduce(
        gram,
        op=dist.ReduceOp.SUM,
        group=state.process_group,
    )

    # gram is symmetric positive semidefinite.
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)

    q = eigenvectors[:, -rank:].contiguous()
    q = q.to(dtype=dtype)

    # Safety orthogonalization after dtype cast.
    q = orthogonalize(q)

    return q


def standard_powersgd_step(
    matrix: torch.Tensor,
    q_memory: torch.Tensor,
    state: FSDPPowerSGDPlusEFState,
):
    """
    Correct PowerSGD step.

    matrix:   [rows, cols]
    q_memory: [cols, r], orthonormal basis from previous step / SVD refresh

    p = M q
    all_reduce(p)
    p = orthogonalize(p)

    q = M^T p
    all_reduce(q)

    approx = p q^T

    Important:
        q must NOT be orthogonalized before reconstruction,
        otherwise the scale of the approximation is destroyed.
    """
    p = matrix.matmul(q_memory)

    dist.all_reduce(
        p,
        op=dist.ReduceOp.SUM,
        group=state.process_group,
    )
    p.div_(state.world_size)

    p = orthogonalize(p)

    q = matrix.t().matmul(p)

    dist.all_reduce(
        q,
        op=dist.ReduceOp.SUM,
        group=state.process_group,
    )
    q.div_(state.world_size)

    approx = p.matmul(q.t())

    # Store orthogonalized q only for the next iteration's projection basis.
    q_next = orthogonalize(q)

    return approx, q_next

def all_reduce_average(tensor: torch.Tensor, state: FSDPPowerSGDPlusEFState):
    out = tensor.contiguous()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=state.process_group)
    out.div_(state.world_size)
    return out


def powersgd_plus_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDPlusEFState,
    hook_id: int,
):
    dtype = tensor.dtype
    device = tensor.device
    elem_size = tensor.element_size()
    numel = tensor.numel()

    r = state.matrix_approximation_rank
    cols = state.block_cols

    main_numel = (numel // cols) * cols
    tail_numel = numel - main_numel

    reconstructed = torch.empty_like(tensor)
    compressed_bytes = 0

    used_svd = False
    rows = 0

    if main_numel > 0:
        matrix = tensor[:main_numel].view(-1, cols)
        rows = matrix.shape[0]

        original_main_elems = rows * cols
        compressed_elems = r * (rows + cols)

        use_compression = (
            state.should_compress()
            and compressed_elems * state.min_compression_rate < original_main_elems
        )

        if use_compression:
            if state.should_svd_refresh():
                q = compute_global_svd_q(
                    matrix=matrix,
                    rank=r,
                    state=state,
                )
                used_svd = True
            else:
                q = state.get_q_memory(
                    hook_id=hook_id,
                    cols=cols,
                    rank=r,
                    device=device,
                    dtype=dtype,
                )

            approx, q_new = standard_powersgd_step(
                matrix=matrix,
                q_memory=q,
                state=state,
            )

            # rel_error = (
            #     (matrix - approx).norm()
            #     / matrix.norm()
            # )
            # print(rel_error)

            state.save_q_memory(hook_id, q_new)

            reconstructed[:main_numel].copy_(approx.reshape(-1))

            compressed_bytes += (rows * r + cols * r) * elem_size

            # SVD safeguard communication accounting:
            # q refresh requires communicating Gram matrix, logically cols x cols fp32.
            # This is not compressed gradient traffic, but it is real communication.
            if used_svd:
                compressed_bytes += cols * cols * 4

        else:
            avg = all_reduce_average(
                matrix.contiguous(),
                state,
            )
            reconstructed[:main_numel].copy_(avg.reshape(-1))
            compressed_bytes += original_main_elems * elem_size

    if tail_numel > 0:
        tail_avg = all_reduce_average(
            tensor[main_numel:].contiguous(),
            state,
        )
        reconstructed[main_numel:].copy_(tail_avg)
        compressed_bytes += tail_numel * elem_size

    state.add_svd_record(
        hook_id=hook_id,
        used_svd=used_svd,
        rows=rows,
        cols=cols,
        rank=r,
    )

    return reconstructed, compressed_bytes


def fsdp_powersgd_plus_error_feedback_hook(
    state: FSDPPowerSGDPlusEFState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    op_name = (
        "powersgd_plus_ef_reduce_scatter"
        if output is not None
        else "powersgd_plus_ef_all_reduce"
    )

    hook_id = state.hook_call_id
    state.hook_call_id += 1

    original_bytes = grad.numel() * grad.element_size()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if state.use_error_feedback:
        residual = state.get_residual(grad)
        # print(residual.abs().sum())
        corrected_grad = torch.empty_like(grad)
        corrected_grad.copy_(grad)
        corrected_grad.add_(0.5*residual)

        reconstructed, compressed_bytes = powersgd_plus_compress_reconstruct(
            tensor=corrected_grad,
            state=state,
            hook_id=hook_id,
        )

        # rel_error = (
        #     (corrected_grad - reconstructed).norm()
        #     / corrected_grad.norm()
        # )
        # print(rel_error)

        residual.copy_(corrected_grad)
        residual.sub_(reconstructed)

        del corrected_grad

    else:
        reconstructed, compressed_bytes = powersgd_plus_compress_reconstruct(
            tensor=grad,
            state=state,
            hook_id=hook_id,
        )

    if output is None:
        grad.copy_(reconstructed)
    else:
        assert grad.ndim == 1
        assert output.ndim == 1

        shard_size = output.numel()
        start = state.rank * shard_size
        end = start + shard_size

        output.copy_(reconstructed.view(-1)[start:end].to(dtype=output.dtype))

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        op_name=op_name,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        latency_ms=latency_ms,
    )

    del reconstructed


def register_fsdp_powersgd_plus_error_feedback_hook(
    model,
    rank: int,
    world_size: int,
    matrix_approximation_rank: int = 2,
    block_cols: int = 4096,
    min_compression_rate: float = 2.0,
    svd_refresh_period: int = 50,
    start_compression_iter: int = 10,
    use_error_feedback: bool = True,
):
    state = FSDPPowerSGDPlusEFState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
        svd_refresh_period=svd_refresh_period,
        start_compression_iter=start_compression_iter,
        use_error_feedback=use_error_feedback,
    )

    model.register_comm_hook(
        state,
        fsdp_powersgd_plus_error_feedback_hook,
    )

    return state