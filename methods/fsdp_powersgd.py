import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


def format_bytes(n: int) -> str:
    if n == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}" if i > 0 else f"{int(x)}B"


@dataclass
class FSDPPowerSGDState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 1
    block_cols: int = 4096
    min_compression_rate: float = 2.0
    use_error_feedback: bool = True
    eps: float = 1e-8

    residuals: dict = field(default_factory=dict)
    hook_idx: int = 0
    summary: dict = field(default_factory=dict)

    start_compression_iter: int = 0
    iter: int = 0

    def reset(self):
        self.summary = {}
        self.hook_idx = 0
        self.iter += 1

    def get_residual(self, tensor: torch.Tensor):
        key = self.hook_idx
        self.hook_idx += 1

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

    def add_record(
        self,
        op_name: str,
        original_bytes: int,
        compressed_bytes: int,
        latency_ms: float,
    ):
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


def _orthogonalize(mat: torch.Tensor, eps: float):
    # QR is stable but can be slower; for rank <= 4 it is acceptable.
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def _powersgd_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDState,
):
    """
    PowerSGD-like low-rank compression for a flattened FSDP gradient buffer.

    We reshape most of the flat tensor into [rows, block_cols].
    Tail that does not fit is synchronized with normal all_reduce.

    Returns:
        reconstructed full averaged tensor
        logical compressed bytes
    """
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

    if main_numel > 0:
        matrix = tensor[:main_numel].view(-1, cols)
        rows = matrix.shape[0]

        original_main_elems = rows * cols
        compressed_elems = r * (rows + cols)

        if compressed_elems * state.min_compression_rate < original_main_elems:
            # Q: [cols, r]
            # Use deterministic random Q per hook call shape.
            q = torch.randn(cols, r, device=device, dtype=dtype)
            q = _orthogonalize(q, state.eps)

            # P = M Q
            p = matrix.matmul(q)

            dist.all_reduce(p, op=dist.ReduceOp.SUM, group=state.process_group)
            p.div_(state.world_size)

            p = _orthogonalize(p, state.eps)

            # Q = M^T P
            q = matrix.t().matmul(p)

            dist.all_reduce(q, op=dist.ReduceOp.SUM, group=state.process_group)
            q.div_(state.world_size)

            approx = p.matmul(q.t())
            reconstructed[:main_numel].copy_(approx.reshape(-1))

            compressed_bytes += (p.numel() + q.numel()) * elem_size
        else:
            tmp = matrix.contiguous()
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=state.process_group)
            tmp.div_(state.world_size)
            reconstructed[:main_numel].copy_(tmp.reshape(-1))
            compressed_bytes += original_main_elems * elem_size

    if tail_numel > 0:
        tail = tensor[main_numel:].contiguous()
        dist.all_reduce(tail, op=dist.ReduceOp.SUM, group=state.process_group)
        tail.div_(state.world_size)
        reconstructed[main_numel:].copy_(tail)
        compressed_bytes += tail_numel * elem_size

    return reconstructed, compressed_bytes


def fsdp_powersgd_hook(
    state: FSDPPowerSGDState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    """
    FSDP communication hook with PowerSGD-like gradient compression.

    If use_error_feedback=True:
        u_t = g_t + e_t
        c_t = PowerSGD(u_t)
        e_{t+1} = u_t - c_t

    For FSDP FULL_SHARD:
        output receives this rank's shard of reconstructed averaged gradient.
    """
    op_name = "powersgd_reduce_scatter" if output is not None else "powersgd_all_reduce"

    original_bytes = grad.numel() * grad.element_size()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # WARMUP
    if state.iter < state.start_compression_iter:
        if output is None:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=state.process_group)
            grad.div_(state.world_size)
            compressed_bytes = original_bytes
        else:
            dist.reduce_scatter_tensor(
                output,
                grad,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            output.div_(state.world_size)
            compressed_bytes = original_bytes

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record(
            op_name="warmup_no_compression",
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            latency_ms=latency_ms,
        )
        return
    # WARMUP

    if state.use_error_feedback:
        residual = state.get_residual(grad)

        compensated = torch.empty_like(grad)
        compensated.copy_(grad)
        compensated.add_(residual)

        reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
            compensated,
            state,
        )

        residual.copy_(compensated)
        residual.sub_(reconstructed)

        del compensated
    else:
        reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
            grad,
            state,
        )

    if output is None:
        grad.copy_(reconstructed)
    else:
        chunks = reconstructed.chunk(state.world_size)
        output.copy_(chunks[state.rank].contiguous())

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        op_name=op_name,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        latency_ms=latency_ms,
    )

    del reconstructed


def register_fsdp_powersgd_hook(
    model,
    rank: int,
    world_size: int,
    matrix_approximation_rank: int = 1,
    block_cols: int = 4096,
    min_compression_rate: float = 2.0,
    use_error_feedback: bool = True,
    start_compression_iter: int = 0,
):
    state = FSDPPowerSGDState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
        use_error_feedback=use_error_feedback,
        start_compression_iter=start_compression_iter
    )

    model.register_comm_hook(state, fsdp_powersgd_hook)
    return state