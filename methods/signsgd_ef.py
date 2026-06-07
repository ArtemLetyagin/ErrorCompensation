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
class FSDPScaledSignEFState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    eps: float = 1e-8

    # residuals are indexed by hook call order inside one optimizer iteration
    residuals: dict = field(default_factory=dict)
    hook_idx: int = 0

    summary: dict = field(default_factory=dict)

    def reset(self):
        # Called once per outer training iteration.
        # Do not clear residuals: they are the error-feedback memory.
        self.summary = {}
        self.hook_idx = 0

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


def _scaled_sign_inplace(
    x: torch.Tensor,
    eps: float,
):
    """
    Returns:
        sign_fp: tensor with values -1 / +1 in x.dtype
        scale: scalar tensor in x.dtype

    Important:
        This function does not allocate int8 signs.
        It creates one fp tensor because NCCL all_reduce does not support int8
        as a practical compression transport in this simple hook.

    Logical communication size is still logged as 1 bit/sign-style payload,
    but physical NCCL traffic is fp32/fp16 unless custom packing is implemented.
    """
    scale = x.abs().mean().clamp_min(eps)

    sign_fp = torch.empty_like(x)
    sign_fp.copy_(x)
    sign_fp.sign_()

    # torch.sign(0) = 0, but sign compression usually maps zeros to +1.
    sign_fp.masked_fill_(sign_fp == 0, 1.0)

    return sign_fp, scale


def fsdp_scaled_sign_ef_hook(
    state: FSDPScaledSignEFState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    """
    FSDP communication hook for scaled sign compression with error feedback.

    Algorithm:
        u_t = g_t + e_t
        scale_t = mean(abs(u_t))
        c_t = scale_t * sign(u_t)
        e_{t+1} = u_t - c_t

    Distributed averaging:
        Each rank sends sign(u_t) and scale_t.
        The reconstructed gradient is:
            mean_i [scale_i * sign_i]

    Notes:
        - This is a communication-hook baseline, not a replacement for AdamW.
        - Optimizer can remain AdamW.
        - For FSDP FULL_SHARD, output is the local shard expected by FSDP.
    """
    op_name = (
        "scaled_sign_ef_reduce_scatter"
        if output is not None
        else "scaled_sign_ef_all_reduce"
    )

    original_bytes = grad.numel() * grad.element_size()

    # Logical compressed size:
    # sign per element + one fp32 scale.
    # If you want true bit-level accounting, use ceil(numel / 8) + 4.
    # If you want int8-style accounting, use numel + 4.
    compressed_bytes = (grad.numel() + 7) // 8 + 4

    residual = state.get_residual(grad)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # compensated = grad + residual
    # One full-size temporary is needed to update EF correctly.
    compensated = torch.empty_like(grad)
    compensated.copy_(grad)
    compensated.add_(residual)

    sign_fp, scale = _scaled_sign_inplace(compensated, state.eps)

    # Average signs and scales across ranks.
    dist.all_reduce(sign_fp, op=dist.ReduceOp.SUM, group=state.process_group)
    dist.all_reduce(scale, op=dist.ReduceOp.SUM, group=state.process_group)

    sign_fp.div_(state.world_size)
    scale.div_(state.world_size)

    # sign_fp now stores reconstructed full averaged gradient estimate.
    sign_fp.mul_(scale)

    # residual = compensated - reconstructed
    residual.copy_(compensated)
    residual.sub_(sign_fp)

    if output is None:
        grad.copy_(sign_fp)
    else:
        # FSDP expects only this rank's shard.
        chunks = sign_fp.chunk(state.world_size)
        output.copy_(chunks[state.rank].contiguous())

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        op_name=op_name,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        latency_ms=latency_ms,
    )

    # Help allocator release temporaries earlier.
    del compensated
    del sign_fp


def register_fsdp_scaled_sign_ef_hook(
    model,
    rank: int,
    world_size: int,
    eps: float = 1e-8,
):
    state = FSDPScaledSignEFState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        eps=eps,
    )

    model.register_comm_hook(state, fsdp_scaled_sign_ef_hook)
    return state