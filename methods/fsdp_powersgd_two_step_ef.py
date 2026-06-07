import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPPowerSGDTwoStepEFState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 2
    block_cols: int = 4096
    min_compression_rate: float = 2.0

    gamma_prev: float = 0.0
    start_compression_iter: int = 1
    eps: float = 1e-8
    seed: int = 1234

    residuals_curr: dict = field(default_factory=dict)
    residuals_prev: dict = field(default_factory=dict)

    hook_call_id: int = 0
    iter_idx: int = 0

    summary: dict = field(default_factory=dict)
    residual_stats: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}
        self.residual_stats = {}
        self.hook_call_id = 0
        self.iter_idx += 1

    def should_compress(self):
        return self.iter_idx >= self.start_compression_iter

    def get_residuals(self, grad: torch.Tensor):
        hook_id = self.hook_call_id
        self.hook_call_id += 1

        curr = self.residuals_curr.get(hook_id)
        prev = self.residuals_prev.get(hook_id)

        if curr is None or curr.shape != grad.shape or curr.device != grad.device:
            curr = torch.zeros_like(grad)

        if prev is None or prev.shape != grad.shape or prev.device != grad.device:
            prev = torch.zeros_like(grad)

        return hook_id, curr, prev

    def save_residuals(self, hook_id, new_curr, old_curr):
        self.residuals_prev[hook_id] = old_curr.detach().clone()
        self.residuals_curr[hook_id] = new_curr.detach().clone()

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

    def add_residual_record(self, hook_id, curr_norm, prev_norm, gamma_prev):
        key = str(hook_id)

        if key not in self.residual_stats:
            self.residual_stats[key] = {
                "count": 0,
                "curr_norm_sum": 0.0,
                "prev_norm_sum": 0.0,
                "curr_norm_avg": 0.0,
                "prev_norm_avg": 0.0,
                "curr_norm_last": 0.0,
                "prev_norm_last": 0.0,
                "gamma_prev": gamma_prev,
            }

        rec = self.residual_stats[key]
        rec["count"] += 1
        rec["curr_norm_sum"] += float(curr_norm)
        rec["prev_norm_sum"] += float(prev_norm)
        rec["curr_norm_avg"] = rec["curr_norm_sum"] / rec["count"]
        rec["prev_norm_avg"] = rec["prev_norm_sum"] / rec["count"]
        rec["curr_norm_last"] = float(curr_norm)
        rec["prev_norm_last"] = float(prev_norm)


def _sync_cuda_if_needed(tensor: torch.Tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _orthogonalize(mat: torch.Tensor):
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def _make_shared_random_q(
    rows: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    state: FSDPPowerSGDTwoStepEFState,
    hook_id: int,
):
    q_dtype = torch.float32

    gen = torch.Generator(device=device)
    gen.manual_seed(
        int(state.seed)
        + int(state.iter_idx) * 1000003
        + int(hook_id) * 9176
    )

    q = torch.randn(
        rows,
        rank,
        device=device,
        dtype=q_dtype,
        generator=gen,
    )

    q = _orthogonalize(q)

    if dtype != torch.float32:
        q = q.to(dtype=dtype)

    return q.contiguous()


def _powersgd_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDTwoStepEFState,
    hook_id: int,
):
    original_dtype = tensor.dtype
    device = tensor.device
    elem_size = tensor.element_size()
    numel = tensor.numel()

    block_cols = state.block_cols
    main_numel = (numel // block_cols) * block_cols
    tail_numel = numel - main_numel

    reconstructed = torch.empty_like(tensor)
    compressed_bytes = 0

    if main_numel > 0:
        matrix = tensor[:main_numel].view(-1, block_cols)
        rows, cols = matrix.shape

        rank = min(state.matrix_approximation_rank, rows, cols)

        original_main_elems = rows * cols
        compressed_elems = rank * (rows + cols)

        use_compression = (
            state.should_compress()
            and rank > 0
            and compressed_elems * state.min_compression_rate < original_main_elems
        )

        if use_compression:
            compute_dtype = torch.float32
            matrix_compute = matrix.to(compute_dtype)

            q = _make_shared_random_q(
                rows=cols,
                rank=rank,
                device=device,
                dtype=compute_dtype,
                state=state,
                hook_id=hook_id,
            )

            p = matrix_compute.matmul(q)

            dist.all_reduce(
                p,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            p.div_(state.world_size)

            p = _orthogonalize(p)

            q = matrix_compute.t().matmul(p)

            dist.all_reduce(
                q,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            q.div_(state.world_size)

            approx = p.matmul(q.t())

            reconstructed[:main_numel].copy_(
                approx.reshape(-1).to(dtype=original_dtype)
            )

            compressed_bytes += (p.numel() + q.numel()) * elem_size

        else:
            tmp = matrix.contiguous()
            dist.all_reduce(
                tmp,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            tmp.div_(state.world_size)

            reconstructed[:main_numel].copy_(tmp.reshape(-1))
            compressed_bytes += original_main_elems * elem_size

    if tail_numel > 0:
        tail = tensor[main_numel:].contiguous()

        dist.all_reduce(
            tail,
            op=dist.ReduceOp.SUM,
            group=state.process_group,
        )
        tail.div_(state.world_size)

        reconstructed[main_numel:].copy_(tail)
        compressed_bytes += tail_numel * elem_size

    return reconstructed, compressed_bytes


def fsdp_powersgd_two_step_error_feedback_hook(
    state: FSDPPowerSGDTwoStepEFState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    op_name = (
        "powersgd_two_step_ef_reduce_scatter"
        if output is not None
        else "powersgd_two_step_ef_all_reduce"
    )

    original_bytes = grad.numel() * grad.element_size()

    hook_id, curr_residual, prev_residual = state.get_residuals(grad)

    _sync_cuda_if_needed(grad)
    t0 = time.perf_counter()

    corrected_grad = grad.detach().clone()
    corrected_grad.add_(curr_residual)

    if state.gamma_prev != 0.0:
        corrected_grad.add_(prev_residual, alpha=state.gamma_prev)

    reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
        corrected_grad,
        state,
        hook_id,
    )

    new_residual = corrected_grad - reconstructed

    state.save_residuals(
        hook_id=hook_id,
        new_curr=new_residual,
        old_curr=curr_residual,
    )

    state.add_residual_record(
        hook_id=hook_id,
        curr_norm=torch.norm(curr_residual.float()).detach().item(),
        prev_norm=torch.norm(prev_residual.float()).detach().item(),
        gamma_prev=state.gamma_prev,
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

    _sync_cuda_if_needed(grad)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        op_name=op_name,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        latency_ms=latency_ms,
    )


def register_fsdp_powersgd_two_step_error_feedback_hook(
    model,
    rank,
    world_size,
    matrix_approximation_rank=2,
    block_cols=4096,
    min_compression_rate=2.0,
    gamma_prev=0.0,
    start_compression_iter=1,
    seed=1234,
):
    state = FSDPPowerSGDTwoStepEFState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
        gamma_prev=gamma_prev,
        start_compression_iter=start_compression_iter,
        seed=seed,
    )

    model.register_comm_hook(
        state,
        fsdp_powersgd_two_step_error_feedback_hook,
    )

    return state