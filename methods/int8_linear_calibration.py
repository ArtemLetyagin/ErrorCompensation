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
class FSDPInt8LinearCalibrationCommState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    summary: dict = field(default_factory=dict)

    # hook_id -> {"a": tensor, "b": tensor}
    calibrations: dict = field(default_factory=dict)

    hook_call_id: int = 0
    eps: float = 1e-8

    def reset(self):
        self.summary = {}
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


def quantize_int8_symmetric(x: torch.Tensor):
    x = x.contiguous()

    max_abs = x.abs().max()
    scale = max_abs / 127.0

    if scale.item() == 0.0:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)

    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)

    return q, scale.float()


def dequantize_int8_symmetric(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale


def compute_linear_calibration(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Fits y ~= a * x + b.

    x: dequantized tensor
    y: original corrected tensor

    Returned calibration is used as:
        calibrated = a * deq + b
    """
    x_f = x.float().view(-1)
    y_f = y.float().view(-1)

    x_mean = x_f.mean()
    y_mean = y_f.mean()

    var_x = ((x_f - x_mean) ** 2).mean()
    cov_xy = ((x_f - x_mean) * (y_f - y_mean)).mean()

    a = cov_xy / (var_x + eps)
    b = y_mean - a * x_mean

    return a.detach(), b.detach()


def apply_linear_calibration(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    return a.to(x.device) * x.float() + b.to(x.device)


def fsdp_int8_linear_calibration_hook(
    state: FSDPInt8LinearCalibrationCommState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    world_size = state.world_size
    rank = state.rank

    hook_id = state.hook_call_id
    state.hook_call_id += 1

    calibration = state.calibrations.get(hook_id)

    q, scale = quantize_int8_symmetric(grad)
    deq = dequantize_int8_symmetric(q, scale)

    # Apply previous linear calibration if available:
    # deq_corr = a * deq + b
    if calibration is not None:
        a = calibration["a"]
        b = calibration["b"]
        deq_corr = apply_linear_calibration(deq, a, b)
    else:
        deq_corr = deq

    # Update calibration for the next call:
    # grad ~= a * deq + b
    a_new, b_new = compute_linear_calibration(
        x=deq,
        y=grad,
        eps=state.eps,
    )

    state.calibrations[hook_id] = {
        "a": a_new,
        "b": b_new,
    }

    # -------------------------
    # NO_SHARD / all_reduce case
    # -------------------------
    if output is None:
        q_list = [torch.empty_like(q) for _ in range(world_size)]
        scale_list = [torch.empty_like(scale) for _ in range(world_size)]

        # Each rank also needs to share its calibration parameters.
        a_tensor = state.calibrations[hook_id]["a"].to(
            device=grad.device,
            dtype=torch.float32,
        )
        b_tensor = state.calibrations[hook_id]["b"].to(
            device=grad.device,
            dtype=torch.float32,
        )

        a_list = [torch.empty_like(a_tensor) for _ in range(world_size)]
        b_list = [torch.empty_like(b_tensor) for _ in range(world_size)]

        msg_size_bytes_per_rank = (
            q.numel() * q.element_size()
            + scale.numel() * scale.element_size()
            + a_tensor.numel() * a_tensor.element_size()
            + b_tensor.numel() * b_tensor.element_size()
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_gather(q_list, q, group=state.process_group)
        dist.all_gather(scale_list, scale, group=state.process_group)
        dist.all_gather(a_list, a_tensor, group=state.process_group)
        dist.all_gather(b_list, b_tensor, group=state.process_group)

        reduced = torch.zeros_like(grad)

        for q_i, scale_i, a_i, b_i in zip(q_list, scale_list, a_list, b_list):
            deq_i = dequantize_int8_symmetric(q_i, scale_i)
            deq_i = apply_linear_calibration(deq_i, a_i, b_i)
            reduced += deq_i

        grad.copy_(reduced / world_size)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record(
            "int8_linear_calibration_all_gather_all_reduce",
            msg_size_bytes_per_rank * world_size,
            latency_ms,
        )
        return

    # -------------------------
    # FSDP sharded reduce-scatter case
    # -------------------------
    assert grad.ndim == 1
    assert output.ndim == 1

    q_list = [torch.empty_like(q) for _ in range(world_size)]
    scale_list = [torch.empty_like(scale) for _ in range(world_size)]

    a_tensor = state.calibrations[hook_id]["a"].to(
        device=grad.device,
        dtype=torch.float32,
    )
    b_tensor = state.calibrations[hook_id]["b"].to(
        device=grad.device,
        dtype=torch.float32,
    )

    a_list = [torch.empty_like(a_tensor) for _ in range(world_size)]
    b_list = [torch.empty_like(b_tensor) for _ in range(world_size)]

    msg_size_bytes_per_rank = (
        q.numel() * q.element_size()
        + scale.numel() * scale.element_size()
        + a_tensor.numel() * a_tensor.element_size()
        + b_tensor.numel() * b_tensor.element_size()
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dist.all_gather(q_list, q, group=state.process_group)
    dist.all_gather(scale_list, scale, group=state.process_group)
    dist.all_gather(a_list, a_tensor, group=state.process_group)
    dist.all_gather(b_list, b_tensor, group=state.process_group)

    shard_size = output.numel()
    start = rank * shard_size
    end = start + shard_size

    reduced_shard = torch.zeros_like(output)

    for q_i, scale_i, a_i, b_i in zip(q_list, scale_list, a_list, b_list):
        deq_i = dequantize_int8_symmetric(q_i[start:end], scale_i)
        deq_i = apply_linear_calibration(deq_i, a_i, b_i)
        reduced_shard += deq_i

    output.copy_(reduced_shard / world_size)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        "int8_linear_calibration_all_gather_reduce_scatter",
        msg_size_bytes_per_rank * world_size,
        latency_ms,
    )


def register_fsdp_int8_linear_calibration_hook(model, rank, world_size):
    state = FSDPInt8LinearCalibrationCommState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
    )

    model.register_comm_hook(state, fsdp_int8_linear_calibration_hook)
    return state