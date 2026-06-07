from dataclasses import dataclass, field
import torch.distributed as dist
import torch
import time
from .utils import format_bytes


@dataclass
class FSDPSparseTopKAdaptiveEFCommState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    sparsity: float = 0.99

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


def sparsify_topk(x: torch.Tensor, sparsity: float):
    x = x.contiguous().view(-1)

    numel = x.numel()
    keep_ratio = 1.0 - sparsity
    k = max(1, int(numel * keep_ratio))

    _, indices = torch.topk(x.abs(), k, sorted=False)
    values = x[indices]

    indices = indices.to(torch.int32)

    return indices, values, numel


def make_sparse_reconstruction_like(x, indices, values):
    dense = torch.zeros_like(x.view(-1))
    dense.index_copy_(0, indices.long(), values)
    return dense.view_as(x)


def add_sparse_to_dense(
    dense: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    start: int,
    end: int,
):
    mask = (indices >= start) & (indices < end)

    if not mask.any():
        return

    local_indices = indices[mask].long() - start
    local_values = values[mask]

    dense.index_add_(0, local_indices, local_values)


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


def fsdp_sparse_topk_adaptive_error_feedback_hook(
    state: FSDPSparseTopKAdaptiveEFCommState,
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

    corrected_grad = grad + alpha * residual

    indices, values, _ = sparsify_topk(corrected_grad, state.sparsity)

    reconstructed = make_sparse_reconstruction_like(
        corrected_grad,
        indices,
        values,
    )

    state.residuals[hook_id] = (corrected_grad - reconstructed).detach()

    state.add_alpha_record(
        hook_id=hook_id,
        alpha=alpha.detach().float().item(),
        residual_ratio=residual_ratio.detach().float().item(),
    )

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
            "sparse_topk_adaptive_ef_all_gather_all_reduce",
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
        "sparse_topk_adaptive_ef_all_gather_reduce_scatter",
        msg_size_bytes_per_rank * world_size,
        latency_ms,
    )


def register_fsdp_sparse_topk_adaptive_error_feedback_hook(
    model,
    rank,
    world_size,
    sparsity,
    alpha_min=0.25,
    alpha_max=1.25,
    gamma=0.5,
):
    state = FSDPSparseTopKAdaptiveEFCommState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        sparsity=sparsity,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        gamma=gamma,
    )

    model.register_comm_hook(
        state,
        fsdp_sparse_topk_adaptive_error_feedback_hook,
    )

    return state