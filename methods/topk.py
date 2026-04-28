from dataclasses import dataclass, field
import torch.distributed as dist
import torch
import time
from .utils import format_bytes

@dataclass
class FSDPSparseTopKCommState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    sparsity: float = 0.99
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

def sparsify_topk(x: torch.Tensor, sparsity: float):
    x = x.contiguous().view(-1)

    numel = x.numel()
    keep_ratio = 1.0 - sparsity
    k = max(1, int(numel * keep_ratio))

    _, indices = torch.topk(x.abs(), k, sorted=False)
    values = x[indices]

    # int32 достаточно, если tensor меньше 2^31 элементов
    indices = indices.to(torch.int32)

    return indices, values, numel

def add_sparse_to_dense(
    dense: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    start: int,
    end: int,
):
    """
    Adds sparse vector entries from [start, end) into local dense shard.

    dense shape: [end - start]
    indices: global flat indices
    values: values for these indices
    """

    mask = (indices >= start) & (indices < end)

    if not mask.any():
        return

    local_indices = indices[mask].long() - start
    local_values = values[mask]

    dense.index_add_(0, local_indices, local_values)

def fsdp_sparse_topk_comm_hook(
    state: FSDPSparseTopKCommState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    world_size = state.world_size
    rank = state.rank

    indices, values, numel = sparsify_topk(grad, state.sparsity)

    # Передаём:
    # indices int32 + values grad.dtype
    msg_size_bytes_per_rank = (
        indices.numel() * indices.element_size()
        + values.numel() * values.element_size()
    )

    indices_list = [torch.empty_like(indices) for _ in range(world_size)]
    values_list = [torch.empty_like(values) for _ in range(world_size)]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dist.all_gather(indices_list, indices, group=state.process_group)
    dist.all_gather(values_list, values, group=state.process_group)

    # -------------------------
    # NO_SHARD / all_reduce case
    # -------------------------
    if output is None:
        reduced = torch.zeros_like(grad.view(-1))

        for idx_i, val_i in zip(indices_list, values_list):
            reduced.index_add_(0, idx_i.long(), val_i)

        grad.copy_((reduced / world_size).view_as(grad))

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record(
            "sparse_topk_all_gather_all_reduce",
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

    for idx_i, val_i in zip(indices_list, values_list):
        add_sparse_to_dense(
            dense=reduced_shard,
            indices=idx_i,
            values=val_i,
            start=start,
            end=end,
        )

    output.copy_(reduced_shard / world_size)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        "sparse_topk_all_gather_reduce_scatter",
        msg_size_bytes_per_rank * world_size,
        latency_ms,
    )

def register_fsdp_sparse_topk_hook(model, rank, world_size, sparsity):
    state = FSDPSparseTopKCommState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        sparsity=sparsity,
    )

    model.register_comm_hook(state, fsdp_sparse_topk_comm_hook)
    return state