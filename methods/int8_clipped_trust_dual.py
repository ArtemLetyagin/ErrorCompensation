from dataclasses import dataclass, field
import torch.distributed as dist
import torch
import time
from .utils import format_bytes


@dataclass
class FSDPInt8BlockClippedTrustDualEFCommState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    block_size: int = 4096
    percentile: float = 0.999
    eps: float = 1e-8

    # Dual-memory EF params
    mu: float = 0.95          # long residual EMA
    rho: float = 0.9          # trust EMA
    lambda_long: float = 0.5  # long-memory contribution
    trust_min: float = 0.0
    trust_max: float = 1.0

    summary: dict = field(default_factory=dict)
    short_residuals: dict = field(default_factory=dict)
    long_residuals: dict = field(default_factory=dict)
    trust_scores: dict = field(default_factory=dict)
    trust_stats: dict = field(default_factory=dict)
    hook_call_id: int = 0

    def reset(self):
        self.summary = {}
        self.trust_stats = {}
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

    def add_trust_record(self, hook_id, cosine, trust):
        key = str(hook_id)

        if key not in self.trust_stats:
            self.trust_stats[key] = {
                "count": 0,
                "cosine_sum": 0.0,
                "cosine_avg": 0.0,
                "trust_sum": 0.0,
                "trust_avg": 0.0,
                "cosine_last": 0.0,
                "trust_last": 0.0,
            }

        rec = self.trust_stats[key]
        rec["count"] += 1
        rec["cosine_sum"] += float(cosine)
        rec["trust_sum"] += float(trust)

        rec["cosine_avg"] = rec["cosine_sum"] / rec["count"]
        rec["trust_avg"] = rec["trust_sum"] / rec["count"]

        rec["cosine_last"] = float(cosine)
        rec["trust_last"] = float(trust)


def quantize_int8_block_clipped(
    x: torch.Tensor,
    block_size: int = 4096,
    percentile: float = 0.999,
    eps: float = 1e-8,
):
    x_flat = x.contiguous().view(-1)
    numel = x_flat.numel()

    pad = (block_size - numel % block_size) % block_size

    if pad > 0:
        x_flat_padded = torch.cat(
            [
                x_flat,
                torch.zeros(
                    pad,
                    device=x_flat.device,
                    dtype=x_flat.dtype,
                ),
            ],
            dim=0,
        )
    else:
        x_flat_padded = x_flat

    blocks = x_flat_padded.view(-1, block_size)
    abs_blocks = blocks.abs().float()

    if percentile >= 1.0:
        thresholds = abs_blocks.max(dim=1).values
    else:
        thresholds = torch.quantile(
            abs_blocks,
            q=percentile,
            dim=1,
        )

    thresholds = torch.clamp(thresholds, min=eps)

    clipped = torch.clamp(
        blocks.float(),
        min=-thresholds[:, None],
        max=thresholds[:, None],
    )

    scales = thresholds / 127.0
    scales = torch.clamp(scales, min=eps)

    q = torch.round(clipped / scales[:, None])
    q = torch.clamp(q, -127, 127).to(torch.int8)

    return q.view(-1), scales.to(torch.float32), numel, pad


def dequantize_int8_block_clipped(
    q: torch.Tensor,
    scales: torch.Tensor,
    numel: int,
    shape,
    dtype: torch.dtype,
    block_size: int = 4096,
):
    q_blocks = q.view(-1, block_size)
    x = q_blocks.float() * scales[:, None]
    x = x.view(-1)[:numel]
    return x.to(dtype).view(shape)


def compute_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
):
    x_f = x.float().view(-1)
    y_f = y.float().view(-1)

    x_norm = torch.norm(x_f)
    y_norm = torch.norm(y_f)

    cosine = torch.dot(x_f, y_f) / (x_norm * y_norm + eps)
    return cosine


def fsdp_int8_block_clipped_trust_dual_error_feedback_hook(
    state: FSDPInt8BlockClippedTrustDualEFCommState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    world_size = state.world_size
    rank = state.rank

    hook_id = state.hook_call_id
    state.hook_call_id += 1

    short_residual = state.short_residuals.get(hook_id)
    long_residual = state.long_residuals.get(hook_id)
    trust = state.trust_scores.get(hook_id)

    if short_residual is None or short_residual.shape != grad.shape:
        short_residual = torch.zeros_like(grad)

    if long_residual is None or long_residual.shape != grad.shape:
        long_residual = torch.zeros_like(grad)

    if trust is None:
        trust = torch.tensor(
            0.0,
            device=grad.device,
            dtype=torch.float32,
        )

    cosine = compute_cosine_similarity(
        grad,
        long_residual,
        eps=state.eps,
    )

    current_trust = torch.clamp(cosine, min=0.0, max=1.0)
    trust = state.rho * trust + (1.0 - state.rho) * current_trust
    trust = torch.clamp(trust, min=state.trust_min, max=state.trust_max)

    corrected_grad = (
        grad
        + short_residual
        + state.lambda_long * trust.to(dtype=grad.dtype) * long_residual
    )

    q, scales, numel, pad = quantize_int8_block_clipped(
        corrected_grad,
        block_size=state.block_size,
        percentile=state.percentile,
        eps=state.eps,
    )

    reconstructed = dequantize_int8_block_clipped(
        q=q,
        scales=scales,
        numel=numel,
        shape=grad.shape,
        dtype=grad.dtype,
        block_size=state.block_size,
    )

    error = (corrected_grad - reconstructed).detach()

    state.short_residuals[hook_id] = error

    state.long_residuals[hook_id] = (
        state.mu * long_residual + (1.0 - state.mu) * error
    ).detach()

    state.trust_scores[hook_id] = trust.detach()

    state.add_trust_record(
        hook_id=hook_id,
        cosine=cosine.detach().float().item(),
        trust=trust.detach().float().item(),
    )

    q_list = [torch.empty_like(q) for _ in range(world_size)]
    scales_list = [torch.empty_like(scales) for _ in range(world_size)]

    msg_size_bytes_per_rank = (
        q.numel() * q.element_size()
        + scales.numel() * scales.element_size()
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dist.all_gather(q_list, q, group=state.process_group)
    dist.all_gather(scales_list, scales, group=state.process_group)

    # -------------------------
    # NO_SHARD / all_reduce case
    # -------------------------
    if output is None:
        reduced = torch.zeros_like(grad)

        for q_i, scales_i in zip(q_list, scales_list):
            deq_i = dequantize_int8_block_clipped(
                q=q_i,
                scales=scales_i,
                numel=numel,
                shape=grad.shape,
                dtype=grad.dtype,
                block_size=state.block_size,
            )
            reduced.add_(deq_i)

        grad.copy_(reduced / world_size)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record(
            "int8_block_clipped_trust_dual_ef_all_gather_all_reduce",
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

    for q_i, scales_i in zip(q_list, scales_list):
        deq_i = dequantize_int8_block_clipped(
            q=q_i,
            scales=scales_i,
            numel=numel,
            shape=grad.shape,
            dtype=grad.dtype,
            block_size=state.block_size,
        )

        reduced_shard.add_(deq_i.view(-1)[start:end].to(dtype=output.dtype))

    output.copy_(reduced_shard / world_size)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        "int8_block_clipped_trust_dual_ef_all_gather_reduce_scatter",
        msg_size_bytes_per_rank * world_size,
        latency_ms,
    )


def register_fsdp_int8_block_clipped_trust_dual_error_feedback_hook(
    model,
    rank,
    world_size,
    block_size=4096,
    percentile=0.999,
    mu=0.95,
    rho=0.9,
    lambda_long=0.5,
    trust_min=0.0,
    trust_max=1.0,
):
    state = FSDPInt8BlockClippedTrustDualEFCommState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        block_size=block_size,
        percentile=percentile,
        mu=mu,
        rho=rho,
        lambda_long=lambda_long,
        trust_min=trust_min,
        trust_max=trust_max,
    )

    model.register_comm_hook(
        state,
        fsdp_int8_block_clipped_trust_dual_error_feedback_hook,
    )

    return state