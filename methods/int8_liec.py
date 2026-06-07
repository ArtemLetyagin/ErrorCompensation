from dataclasses import dataclass, field
import time

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPInt8LIECState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    block_size: int = 4096
    percentile: float = 0.999
    eps: float = 1e-8

    global_errors: dict = field(default_factory=dict)
    hook_call_id: int = 0
    summary: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}
        self.hook_call_id = 0

    def get_global_error(self, grad: torch.Tensor, hook_id: int):
        err = self.global_errors.get(hook_id)

        if (
            err is None
            or err.shape != grad.shape
            or err.device != grad.device
            or err.dtype != grad.dtype
        ):
            err = torch.zeros_like(grad)
            self.global_errors[hook_id] = err

        return err

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
        x_flat = torch.cat(
            [
                x_flat,
                torch.zeros(pad, device=x.device, dtype=x.dtype),
            ],
            dim=0,
        )

    blocks = x_flat.view(-1, block_size)
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

    scales = torch.clamp(thresholds / 127.0, min=eps)

    q = torch.round(clipped / scales[:, None])
    q = torch.clamp(q, -127, 127).to(torch.int8)

    return q.view(-1), scales.to(torch.float32), numel


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
    return x.to(dtype=dtype).view(shape)


def fsdp_int8_liec_hook(
    state: FSDPInt8LIECState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    """
    LIEC-style bidirectional compression for FSDP.

    Worker-side:
        p_i = C(g_i)
        local_error_i = g_i - p_i

    Server/global emulation:
        avg_p = mean_i p_i
        corrected_global = avg_p + global_error
        p = C(corrected_global)
        global_error = corrected_global - p

    Immediate local compensation:
        update_i = p + local_error_i
                 = p - p_i + g_i
    """
    world_size = state.world_size
    rank = state.rank

    hook_id = state.hook_call_id
    state.hook_call_id += 1

    original_bytes = grad.numel() * grad.element_size()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # -------------------------
    # 1. Worker-side compression
    # -------------------------
    q_local, scales_local, numel = quantize_int8_block_clipped(
        grad,
        block_size=state.block_size,
        percentile=state.percentile,
        eps=state.eps,
    )

    p_local = dequantize_int8_block_clipped(
        q=q_local,
        scales=scales_local,
        numel=numel,
        shape=grad.shape,
        dtype=grad.dtype,
        block_size=state.block_size,
    )

    local_error = grad - p_local

    # -------------------------
    # 2. Gather compressed worker messages
    # -------------------------
    q_list = [torch.empty_like(q_local) for _ in range(world_size)]
    scales_list = [torch.empty_like(scales_local) for _ in range(world_size)]

    dist.all_gather(q_list, q_local, group=state.process_group)
    dist.all_gather(scales_list, scales_local, group=state.process_group)

    avg_p = torch.zeros_like(grad)

    for q_i, scales_i in zip(q_list, scales_list):
        p_i = dequantize_int8_block_clipped(
            q=q_i,
            scales=scales_i,
            numel=numel,
            shape=grad.shape,
            dtype=grad.dtype,
            block_size=state.block_size,
        )
        avg_p.add_(p_i)

    avg_p.div_(world_size)

    # -------------------------
    # 3. Server-side/global compression with global error
    # -------------------------
    global_error = state.get_global_error(grad, hook_id)

    corrected_global = avg_p + global_error

    q_global, scales_global, _ = quantize_int8_block_clipped(
        corrected_global,
        block_size=state.block_size,
        percentile=state.percentile,
        eps=state.eps,
    )

    p_global = dequantize_int8_block_clipped(
        q=q_global,
        scales=scales_global,
        numel=numel,
        shape=grad.shape,
        dtype=grad.dtype,
        block_size=state.block_size,
    )

    global_error.copy_(corrected_global - p_global)

    # -------------------------
    # 4. Immediate local error compensation
    # -------------------------
    update = p_global + local_error

    if output is None:
        grad.copy_(update)
    else:
        assert grad.ndim == 1
        assert output.ndim == 1

        shard_size = output.numel()
        start = rank * shard_size
        end = start + shard_size

        output.copy_(update.view(-1)[start:end].to(dtype=output.dtype))

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    msg_size_per_rank = (
        q_local.numel() * q_local.element_size()
        + scales_local.numel() * scales_local.element_size()
        + q_global.numel() * q_global.element_size()
        + scales_global.numel() * scales_global.element_size()
    )

    state.add_record(
        "int8_liec_bidirectional",
        msg_size_per_rank * world_size,
        latency_ms,
    )


def register_fsdp_int8_liec_hook(
    model,
    rank,
    world_size,
    block_size=4096,
    percentile=0.999,
):
    state = FSDPInt8LIECState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        block_size=block_size,
        percentile=percentile,
    )

    model.register_comm_hook(state, fsdp_int8_liec_hook)

    return state