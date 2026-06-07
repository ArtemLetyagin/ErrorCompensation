import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPPowerSGDClippedTrustDualEFState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 1
    block_cols: int = 4096
    min_compression_rate: float = 2.0
    eps: float = 1e-8

    # Clipping
    block_size: int = 4096
    percentile: float = 0.999

    # Dual-memory EF
    mu: float = 0.95
    rho: float = 0.9
    lambda_long: float = 0.5
    trust_min: float = 0.0
    trust_max: float = 1.0

    short_residuals: dict = field(default_factory=dict)
    long_residuals: dict = field(default_factory=dict)
    trust_scores: dict = field(default_factory=dict)
    trust_stats: dict = field(default_factory=dict)

    hook_call_id: int = 0
    summary: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}
        self.trust_stats = {}
        self.hook_call_id = 0

    def get_state_buffers(self, grad: torch.Tensor):
        hook_id = self.hook_call_id
        self.hook_call_id += 1

        short = self.short_residuals.get(hook_id)
        long = self.long_residuals.get(hook_id)
        trust = self.trust_scores.get(hook_id)

        if short is None or short.shape != grad.shape or short.device != grad.device:
            short = torch.zeros_like(grad)

        if long is None or long.shape != grad.shape or long.device != grad.device:
            long = torch.zeros_like(grad)

        if trust is None:
            trust = torch.tensor(0.0, device=grad.device, dtype=torch.float32)

        return hook_id, short, long, trust

    def save_state_buffers(self, hook_id, short, long, trust):
        self.short_residuals[hook_id] = short.detach()
        self.long_residuals[hook_id] = long.detach()
        self.trust_scores[hook_id] = trust.detach()

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


def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    x_f = x.float().view(-1)
    y_f = y.float().view(-1)

    x_norm = torch.norm(x_f)
    y_norm = torch.norm(y_f)

    return torch.dot(x_f, y_f) / (x_norm * y_norm + eps)


def block_clip_residual(
    x: torch.Tensor,
    block_size: int = 4096,
    percentile: float = 0.999,
    eps: float = 1e-8,
):
    x_flat = x.contiguous().view(-1)
    numel = x_flat.numel()

    pad = (block_size - numel % block_size) % block_size

    if pad > 0:
        x_padded = torch.cat(
            [
                x_flat,
                torch.zeros(pad, device=x.device, dtype=x.dtype),
            ],
            dim=0,
        )
    else:
        x_padded = x_flat

    blocks = x_padded.view(-1, block_size)
    abs_blocks = blocks.abs().float()

    if percentile >= 1.0:
        thresholds = abs_blocks.max(dim=1).values
    else:
        thresholds = torch.quantile(abs_blocks, q=percentile, dim=1)

    thresholds = torch.clamp(thresholds, min=eps)

    clipped = torch.clamp(
        blocks.float(),
        min=-thresholds[:, None],
        max=thresholds[:, None],
    )

    clipped = clipped.view(-1)[:numel].to(dtype=x.dtype)
    return clipped.view_as(x)


def _orthogonalize(mat: torch.Tensor):
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def _powersgd_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDClippedTrustDualEFState,
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


def fsdp_powersgd_clipped_trust_dual_ef_hook(
    state: FSDPPowerSGDClippedTrustDualEFState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    op_name = (
        "powersgd_clipped_trust_dual_ef_reduce_scatter"
        if output is not None
        else "powersgd_clipped_trust_dual_ef_all_reduce"
    )

    original_bytes = grad.numel() * grad.element_size()

    hook_id, short_residual, long_residual, trust = state.get_state_buffers(grad)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    cosine = compute_cosine_similarity(
        grad,
        long_residual,
        eps=state.eps,
    )

    current_trust = torch.clamp(cosine, min=0.0, max=1.0)
    trust = state.rho * trust + (1.0 - state.rho) * current_trust
    trust = torch.clamp(trust, min=state.trust_min, max=state.trust_max)

    clipped_short = block_clip_residual(
        short_residual,
        block_size=state.block_size,
        percentile=state.percentile,
        eps=state.eps,
    )

    clipped_long = block_clip_residual(
        long_residual,
        block_size=state.block_size,
        percentile=state.percentile,
        eps=state.eps,
    )

    # corrected_grad = (
    #     grad
    #     + clipped_short
    #     + state.lambda_long * trust.to(dtype=grad.dtype) * clipped_long
    # )

    correction = clipped_short + state.lambda_long * trust.to(grad.dtype) * clipped_long

    correction_norm = torch.norm(correction.float())
    grad_norm = torch.norm(grad.float()).clamp_min(state.eps)

    max_corr_norm = 0.25 * grad_norm

    if correction_norm > max_corr_norm:
        correction = correction * (max_corr_norm / correction_norm).to(correction.dtype)

    corrected_grad = grad + correction

    reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
        corrected_grad,
        state,
    )

    error = (corrected_grad - reconstructed).detach()

    new_short = error

    new_long = (
        state.mu * long_residual
        + (1.0 - state.mu) * error
    ).detach()

    state.save_state_buffers(
        hook_id=hook_id,
        short=new_short,
        long=new_long,
        trust=trust,
    )

    state.add_trust_record(
        hook_id=hook_id,
        cosine=cosine.detach().float().item(),
        trust=trust.detach().float().item(),
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

    del corrected_grad
    del reconstructed


def register_fsdp_powersgd_clipped_trust_dual_ef_hook(
    model,
    rank: int,
    world_size: int,
    matrix_approximation_rank: int = 1,
    block_cols: int = 4096,
    min_compression_rate: float = 2.0,
    block_size: int = 4096,
    percentile: float = 0.999,
    mu: float = 0.95,
    rho: float = 0.9,
    lambda_long: float = 0.5,
    trust_min: float = 0.0,
    trust_max: float = 1.0,
):
    state = FSDPPowerSGDClippedTrustDualEFState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
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
        fsdp_powersgd_clipped_trust_dual_ef_hook,
    )

    return state