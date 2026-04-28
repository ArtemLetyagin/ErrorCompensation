from dataclasses import dataclass, field
import torch.distributed as dist
import torch
import time


def format_bytes(n):
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
class FSDPInt8CommState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    summary: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}

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

def quantize_int8_symmetric(x: torch.Tensor):
    x = x.contiguous()

    max_abs = x.abs().max()
    scale = max_abs / 127.0

    if scale == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)

    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)

    return q, scale.float()


def dequantize_int8_symmetric(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale

def fsdp_int8_comm_hook(
    state: FSDPInt8CommState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    world_size = state.world_size
    rank = state.rank

    # NO_SHARD case: emulate compressed all-reduce
    if output is None:
        q, scale = quantize_int8_symmetric(grad)

        q_list = [torch.empty_like(q) for _ in range(world_size)]
        scale_list = [torch.empty_like(scale) for _ in range(world_size)]

        msg_size_bytes = q.numel() * q.element_size() + scale.numel() * scale.element_size()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_gather(q_list, q, group=state.process_group)
        dist.all_gather(scale_list, scale, group=state.process_group)

        reduced = torch.zeros_like(grad)

        for q_i, scale_i in zip(q_list, scale_list):
            reduced += dequantize_int8_symmetric(q_i, scale_i)

        grad.copy_(reduced / world_size)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record("int8_all_gather_all_reduce", msg_size_bytes * world_size, latency_ms)
        return

    # FSDP sharded case: emulate compressed reduce-scatter
    assert grad.ndim == 1
    assert output.ndim == 1

    q, scale = quantize_int8_symmetric(grad)

    q_list = [torch.empty_like(q) for _ in range(world_size)]
    scale_list = [torch.empty_like(scale) for _ in range(world_size)]

    msg_size_bytes_per_rank = q.numel() * q.element_size() + scale.numel() * scale.element_size()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dist.all_gather(q_list, q, group=state.process_group)
    dist.all_gather(scale_list, scale, group=state.process_group)

    shard_size = output.numel()
    start = rank * shard_size
    end = start + shard_size

    reduced_shard = torch.zeros_like(output)

    for q_i, scale_i in zip(q_list, scale_list):
        grad_i = dequantize_int8_symmetric(q_i[start:end], scale_i)
        reduced_shard += grad_i

    output.copy_(reduced_shard / world_size)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        "int8_all_gather_reduce_scatter",
        msg_size_bytes_per_rank * world_size,
        latency_ms,
    )

def register_fsdp_int8_comm_hook(model, rank, world_size):
    state = FSDPInt8CommState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
    )

    model.register_comm_hook(state, fsdp_int8_comm_hook)
    return state