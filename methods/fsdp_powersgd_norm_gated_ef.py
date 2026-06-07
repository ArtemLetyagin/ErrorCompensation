import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPPowerSGDNormGatedEFState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 2
    block_cols: int = 4096
    min_compression_rate: float = 2.0

    tau: float = 0.25
    residual_momentum: float = 0.9
    eps: float = 1e-8

    summary: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    norm_stats: dict = field(default_factory=dict)
    hook_call_id: int = 0

    start_compression_iter: int = 0
    iter: int = 0

    def reset(self):
        self.summary = {}
        self.norm_stats = {}
        self.hook_call_id = 0
        self.iter += 1

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

    def add_norm_record(self, hook_id, grad_norm, residual_norm, scale):
        key = str(hook_id)

        if key not in self.norm_stats:
            self.norm_stats[key] = {
                "count": 0,
                "grad_norm_sum": 0.0,
                "residual_norm_sum": 0.0,
                "scale_sum": 0.0,
                "grad_norm_avg": 0.0,
                "residual_norm_avg": 0.0,
                "scale_avg": 0.0,
                "grad_norm_last": 0.0,
                "residual_norm_last": 0.0,
                "scale_last": 0.0,
            }

        rec = self.norm_stats[key]
        rec["count"] += 1

        rec["grad_norm_sum"] += float(grad_norm)
        rec["residual_norm_sum"] += float(residual_norm)
        rec["scale_sum"] += float(scale)

        rec["grad_norm_avg"] = rec["grad_norm_sum"] / rec["count"]
        rec["residual_norm_avg"] = rec["residual_norm_sum"] / rec["count"]
        rec["scale_avg"] = rec["scale_sum"] / rec["count"]

        rec["grad_norm_last"] = float(grad_norm)
        rec["residual_norm_last"] = float(residual_norm)
        rec["scale_last"] = float(scale)


def _orthogonalize(mat: torch.Tensor):
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def _powersgd_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDNormGatedEFState,
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

    if main_numel > 0:
        matrix = tensor[:main_numel].view(-1, cols)
        rows = matrix.shape[0]

        original_main_elems = rows * cols
        compressed_elems = r * (rows + cols)

        if compressed_elems * state.min_compression_rate < original_main_elems:
            q = torch.randn(cols, r, device=device, dtype=dtype)
            q = _orthogonalize(q)

            p = matrix.matmul(q)

            dist.all_reduce(p, op=dist.ReduceOp.SUM, group=state.process_group)
            p.div_(state.world_size)

            p = _orthogonalize(p)

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


def fsdp_powersgd_norm_gated_error_feedback_hook(
    state: FSDPPowerSGDNormGatedEFState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    op_name = (
        "powersgd_norm_gated_ef_reduce_scatter"
        if output is not None
        else "powersgd_norm_gated_ef_all_reduce"
    )

    rank = state.rank
    world_size = state.world_size

    hook_id = state.hook_call_id
    state.hook_call_id += 1

    residual = state.residuals.get(hook_id)

    if (
        residual is None
        or residual.shape != grad.shape
        or residual.device != grad.device
        or residual.dtype != grad.dtype
    ):
        residual = torch.zeros_like(grad)

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

    grad_norm = torch.norm(grad.float())
    residual_norm = torch.norm(residual.float())
    # print(residual.abs().sum())
    scale = torch.clamp(
        state.tau * grad_norm / (residual_norm + state.eps),
        max=1.0,
    )

    corrected_grad = grad + scale.to(dtype=grad.dtype) * residual

    reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
        corrected_grad,
        state,
    )

    # rel_error = (
    #     (corrected_grad - reconstructed).norm()
    #     / corrected_grad.norm()
    # )
    # print(rel_error)

    error = corrected_grad - reconstructed
    # print(error.abs().sum())
    new_residual = (
        state.residual_momentum * residual
        + (1.0 - state.residual_momentum) * error
    ).detach()

    state.residuals[hook_id] = new_residual

    state.add_norm_record(
        hook_id=hook_id,
        grad_norm=grad_norm.detach().float().item(),
        residual_norm=residual_norm.detach().float().item(),
        scale=scale.detach().float().item(),
    )

    if output is None:
        grad.copy_(reconstructed)
    else:
        assert grad.ndim == 1
        assert output.ndim == 1

        shard_size = output.numel()
        start = rank * shard_size
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

    del corrected_grad
    del reconstructed
    del error


def register_fsdp_powersgd_norm_gated_error_feedback_hook(
    model,
    rank,
    world_size,
    matrix_approximation_rank=2,
    block_cols=4096,
    min_compression_rate=2.0,
    tau=0.25,
    residual_momentum=0.9,
    start_compression_iter: int = 0,
):
    state = FSDPPowerSGDNormGatedEFState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
        tau=tau,
        residual_momentum=residual_momentum,
        start_compression_iter=start_compression_iter
    )

    model.register_comm_hook(
        state,
        fsdp_powersgd_norm_gated_error_feedback_hook,
    )

    return state