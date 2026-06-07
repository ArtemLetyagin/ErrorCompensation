import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPPowerSGDDirectionAwareEFState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 1
    block_cols: int = 4096
    min_compression_rate: float = 2.0

    alpha_min: float = 0.0
    alpha_max: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 0.7
    eps: float = 1e-8

    summary: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    direction_stats: dict = field(default_factory=dict)
    hook_call_id: int = 0

    def reset(self):
        self.summary = {}
        self.direction_stats = {}
        self.hook_call_id = 0

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

    def add_direction_record(self, hook_id, cosine, alpha):
        key = str(hook_id)

        if key not in self.direction_stats:
            self.direction_stats[key] = {
                "count": 0,
                "cosine_sum": 0.0,
                "cosine_avg": 0.0,
                "alpha_sum": 0.0,
                "alpha_avg": 0.0,
                "cosine_last": 0.0,
                "alpha_last": 0.0,
            }

        rec = self.direction_stats[key]
        rec["count"] += 1
        rec["cosine_sum"] += float(cosine)
        rec["alpha_sum"] += float(alpha)
        rec["cosine_avg"] = rec["cosine_sum"] / rec["count"]
        rec["alpha_avg"] = rec["alpha_sum"] / rec["count"]
        rec["cosine_last"] = float(cosine)
        rec["alpha_last"] = float(alpha)


def compute_direction_aware_alpha(
    grad: torch.Tensor,
    residual: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    eps: float,
):
    grad_f = grad.float().view(-1)
    residual_f = residual.float().view(-1)

    grad_norm = torch.norm(grad_f)
    residual_norm = torch.norm(residual_f)

    cosine = torch.dot(grad_f, residual_f) / (grad_norm * residual_norm + eps)
    positive_cosine = torch.clamp(cosine, min=0.0, max=1.0)

    alpha = alpha_min + (alpha_max - alpha_min) * positive_cosine
    alpha = torch.clamp(alpha, min=alpha_min, max=alpha_max)

    return alpha.to(dtype=grad.dtype), cosine.detach()


def _orthogonalize(mat: torch.Tensor):
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def _powersgd_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDDirectionAwareEFState,
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


def fsdp_powersgd_direction_aware_error_feedback_hook(
    state: FSDPPowerSGDDirectionAwareEFState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    op_name = (
        "powersgd_direction_aware_ef_reduce_scatter"
        if output is not None
        else "powersgd_direction_aware_ef_all_reduce"
    )

    world_size = state.world_size
    rank = state.rank

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

    alpha, cosine = compute_direction_aware_alpha(
        grad=grad,
        residual=residual,
        alpha_min=state.alpha_min,
        alpha_max=state.alpha_max,
        eps=state.eps,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Same direction-aware logic as in your int8 method:
    # if grad and residual are co-directed: add residual fully
    # otherwise: add beta * residual, where beta depends on confidence.
    confidence = grad.abs() / (grad.abs() + residual.abs() + state.eps)
    beta = state.beta_min + (state.beta_max - state.beta_min) * confidence

    corrected_grad = torch.where(
        grad * residual >= 0,
        grad + residual,
        grad + beta * residual,
    )

    reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
        corrected_grad,
        state,
    )

    state.residuals[hook_id] = (corrected_grad - reconstructed).detach()

    state.add_direction_record(
        hook_id=hook_id,
        cosine=cosine.detach().float().item(),
        alpha=alpha.detach().float().item(),
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


def register_fsdp_powersgd_direction_aware_error_feedback_hook(
    model,
    rank,
    world_size,
    matrix_approximation_rank=1,
    block_cols=4096,
    min_compression_rate=2.0,
    alpha_min=0.0,
    alpha_max=1.0,
    beta_min=0.1,
    beta_max=0.7,
):
    state = FSDPPowerSGDDirectionAwareEFState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        beta_min=beta_min,
        beta_max=beta_max,
    )

    model.register_comm_hook(
        state,
        fsdp_powersgd_direction_aware_error_feedback_hook,
    )

    return state