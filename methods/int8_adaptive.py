from dataclasses import dataclass, field
import torch.distributed as dist
import torch
import time
from .utils import format_bytes


@dataclass
class FSDPInt8AdaptiveEFCommState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    alpha_min: float = 0.25
    alpha_max: float = 1.25
    gamma: float = 0.5
    eps: float = 1e-8

    summary: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    alpha_stats: dict = field(default_factory=dict)
    hook_call_id: int = 0

    def reset(self):
        self.summary = {}
        self.alpha_stats = {}
        self.hook_call_id = 0

    def add_record(self, op_name, msg_size_bytes, latency_ms):
        key = str(msg_size_bytes)

        if op_name not in self.summary:
            self.summary[op_name] = {}

        if key not in self.summary[op_name]:
            self.summary[op_name][key] = {
                "count": 0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "tput_avg_gbps": 0.0,
                "busbw_avg_gbps": 0.0,
                "msg_size_bytes": msg_size_bytes,
                "msg_size_str": format_bytes(msg_size_bytes),
            }

        rec = self.summary[op_name][key]
        rec["count"] += 1
        rec["total_latency_ms"] += latency_ms
        rec["avg_latency_ms"] = rec["total_latency_ms"] / rec["count"]

        if rec["avg_latency_ms"] > 0:
            rec["tput_avg_gbps"] = (
                msg_size_bytes / (rec["avg_latency_ms"] / 1000.0)
            ) / 1e9
            rec["busbw_avg_gbps"] = rec["tput_avg_gbps"]

    def add_alpha_record(self, hook_id, alpha, residual_ratio):
        key = str(hook_id)

        if key not in self.alpha_stats:
            self.alpha_stats[key] = {
                "count": 0,
                "alpha_sum": 0.0,
                "alpha_avg": 0.0,
                "residual_ratio_sum": 0.0,
                "residual_ratio_avg": 0.0,
                "alpha_last": 0.0,
                "residual_ratio_last": 0.0,
            }

        rec = self.alpha_stats[key]
        rec["count"] += 1
        rec["alpha_sum"] += float(alpha)
        rec["residual_ratio_sum"] += float(residual_ratio)

        rec["alpha_avg"] = rec["alpha_sum"] / rec["count"]
        rec["residual_ratio_avg"] = rec["residual_ratio_sum"] / rec["count"]

        rec["alpha_last"] = float(alpha)
        rec["residual_ratio_last"] = float(residual_ratio)


def quantize_int8_symmetric(x: torch.Tensor, eps: float = 1e-8):
    x_flat = x.contiguous().view(-1)

    max_abs = x_flat.abs().max()
    scale = max_abs / 127.0
    scale = torch.clamp(scale, min=eps)

    q = torch.round(x_flat / scale)
    q = torch.clamp(q, -127, 127).to(torch.int8)

    return q, scale.to(torch.float32), x.numel()


def dequantize_int8_symmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
    shape,
    dtype: torch.dtype,
):
    return (q.float() * scale).to(dtype).view(shape)


def compute_layer_adaptive_alpha(
    grad: torch.Tensor,
    residual: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    gamma: float,
    eps: float,
):
    grad_norm = torch.norm(grad.float())
    residual_norm = torch.norm(residual.float())

    residual_ratio = residual_norm / (grad_norm + eps)

    alpha = alpha_min + gamma * residual_ratio
    alpha = torch.clamp(alpha, min=alpha_min, max=alpha_max)

    return alpha.to(dtype=grad.dtype), residual_ratio.detach()


def fsdp_int8_adaptive_error_feedback_hook(
    state: FSDPInt8AdaptiveEFCommState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    world_size = state.world_size
    rank = state.rank

    hook_id = state.hook_call_id
    state.hook_call_id += 1

    residual = state.residuals.get(hook_id)

    if residual is None or residual.shape != grad.shape:
        residual = torch.zeros_like(grad)

    alpha, residual_ratio = compute_layer_adaptive_alpha(
        grad=grad,
        residual=residual,
        alpha_min=state.alpha_min,
        alpha_max=state.alpha_max,
        gamma=state.gamma,
        eps=state.eps,
    )

    #print(hook_id, alpha.item(), residual_ratio.item())

    corrected_grad = grad + alpha * residual

    q, scale, _ = quantize_int8_symmetric(
        corrected_grad,
        eps=state.eps,
    )

    reconstructed = dequantize_int8_symmetric(
        q=q,
        scale=scale,
        shape=grad.shape,
        dtype=grad.dtype,
    )

    state.residuals[hook_id] = (corrected_grad - reconstructed).detach()

    state.add_alpha_record(
        hook_id=hook_id,
        alpha=alpha.detach().float().item(),
        residual_ratio=residual_ratio.detach().float().item(),
    )

    q_list = [torch.empty_like(q) for _ in range(world_size)]
    scale_list = [torch.empty_like(scale) for _ in range(world_size)]

    msg_size_bytes_per_rank = (
        q.numel() * q.element_size()
        + scale.numel() * scale.element_size()
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dist.all_gather(q_list, q, group=state.process_group)
    dist.all_gather(scale_list, scale, group=state.process_group)

    # -------------------------
    # NO_SHARD / all_reduce case
    # -------------------------
    if output is None:
        reduced = torch.zeros_like(grad)

        for q_i, scale_i in zip(q_list, scale_list):
            deq_i = dequantize_int8_symmetric(
                q=q_i,
                scale=scale_i,
                shape=grad.shape,
                dtype=grad.dtype,
            )
            reduced.add_(deq_i)

        grad.copy_(reduced / world_size)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record(
            "int8_adaptive_ef_all_gather_all_reduce",
            msg_size_bytes_per_rank * world_size,
            latency_ms,
        )
        return

    # -------------------------
    # FSDP sharded reduce-scatter case
    # -------------------------
    assert grad.ndim == 1
    assert output.ndim == 1

    shard_size = output.numel()
    start = rank * shard_size
    end = start + shard_size

    reduced_shard = torch.zeros_like(output)

    for q_i, scale_i in zip(q_list, scale_list):
        deq_i = q_i[start:end].float() * scale_i
        reduced_shard.add_(deq_i.to(dtype=output.dtype))

    output.copy_(reduced_shard / world_size)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        "int8_adaptive_ef_all_gather_reduce_scatter",
        msg_size_bytes_per_rank * world_size,
        latency_ms,
    )


def register_fsdp_int8_adaptive_error_feedback_hook(
    model,
    rank,
    world_size,
    alpha_min=0.25,
    alpha_max=1.25,
    gamma=0.5,
):
    state = FSDPInt8AdaptiveEFCommState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        gamma=gamma,
    )

    model.register_comm_hook(
        state,
        fsdp_int8_adaptive_error_feedback_hook,
    )

    return state