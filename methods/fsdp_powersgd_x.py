import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .utils import format_bytes


@dataclass
class FSDPPowerSGDErrorCompensatedXState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int

    matrix_approximation_rank: int = 2
    block_cols: int = 4096
    min_compression_rate: float = 2.0

    # ErrorCompensatedX parameters
    alpha: float = 0.9
    beta: float = 1.0

    start_compression_iter: int = 1
    seed: int = 1234
    eps: float = 1e-8

    # delta_{t-1}, delta_{t-2}, and filtered compensation e_t
    delta_prev: dict = field(default_factory=dict)
    delta_prev2: dict = field(default_factory=dict)
    error_memory: dict = field(default_factory=dict)

    hook_call_id: int = 0
    iter_idx: int = 0

    summary: dict = field(default_factory=dict)
    residual_stats: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}
        self.residual_stats = {}
        self.hook_call_id = 0
        self.iter_idx += 1

    def should_compress(self):
        return self.iter_idx >= self.start_compression_iter

    def get_alpha_t(self):
        return float(self.alpha)

    def get_alpha_prev(self):
        return float(self.alpha)

    def get_alpha_prev2(self):
        return float(self.alpha)

    def get_error_state(self, grad: torch.Tensor):
        hook_id = self.hook_call_id
        self.hook_call_id += 1

        d1 = self.delta_prev.get(hook_id)
        d2 = self.delta_prev2.get(hook_id)
        mem = self.error_memory.get(hook_id)

        if d1 is None or d1.shape != grad.shape or d1.device != grad.device:
            d1 = torch.zeros_like(grad)

        if d2 is None or d2.shape != grad.shape or d2.device != grad.device:
            d2 = torch.zeros_like(grad)

        if mem is None or mem.shape != grad.shape or mem.device != grad.device:
            mem = torch.zeros_like(grad)

        return hook_id, d1, d2, mem

    def save_error_state(self, hook_id, delta_t, delta_prev, error_memory):
        self.delta_prev2[hook_id] = delta_prev.detach().clone()
        self.delta_prev[hook_id] = delta_t.detach().clone()
        self.error_memory[hook_id] = error_memory.detach().clone()

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

    def add_residual_record(self, hook_id, delta_prev_norm, delta_prev2_norm, memory_norm):
        key = str(hook_id)

        if key not in self.residual_stats:
            self.residual_stats[key] = {
                "count": 0,
                "delta_prev_norm_sum": 0.0,
                "delta_prev2_norm_sum": 0.0,
                "memory_norm_sum": 0.0,
                "delta_prev_norm_avg": 0.0,
                "delta_prev2_norm_avg": 0.0,
                "memory_norm_avg": 0.0,
                "delta_prev_norm_last": 0.0,
                "delta_prev2_norm_last": 0.0,
                "memory_norm_last": 0.0,
                "alpha": self.alpha,
                "beta": self.beta,
            }

        rec = self.residual_stats[key]
        rec["count"] += 1

        rec["delta_prev_norm_sum"] += float(delta_prev_norm)
        rec["delta_prev2_norm_sum"] += float(delta_prev2_norm)
        rec["memory_norm_sum"] += float(memory_norm)

        rec["delta_prev_norm_avg"] = rec["delta_prev_norm_sum"] / rec["count"]
        rec["delta_prev2_norm_avg"] = rec["delta_prev2_norm_sum"] / rec["count"]
        rec["memory_norm_avg"] = rec["memory_norm_sum"] / rec["count"]

        rec["delta_prev_norm_last"] = float(delta_prev_norm)
        rec["delta_prev2_norm_last"] = float(delta_prev2_norm)
        rec["memory_norm_last"] = float(memory_norm)


def _sync_cuda_if_needed(tensor: torch.Tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _orthogonalize(mat: torch.Tensor):
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def _make_shared_random_q(rows, rank, device, state, hook_id):
    gen = torch.Generator(device=device)
    gen.manual_seed(
        int(state.seed)
        + int(state.iter_idx) * 1000003
        + int(hook_id) * 9176
    )

    q = torch.randn(
        rows,
        rank,
        device=device,
        dtype=torch.float32,
        generator=gen,
    )

    return _orthogonalize(q)


def _powersgd_compress_reconstruct(
    tensor: torch.Tensor,
    state: FSDPPowerSGDErrorCompensatedXState,
    hook_id: int,
):
    original_dtype = tensor.dtype
    device = tensor.device
    elem_size = tensor.element_size()
    numel = tensor.numel()

    block_cols = state.block_cols
    main_numel = (numel // block_cols) * block_cols
    tail_numel = numel - main_numel

    reconstructed = torch.empty_like(tensor)
    compressed_bytes = 0

    if main_numel > 0:
        matrix = tensor[:main_numel].view(-1, block_cols)
        rows, cols = matrix.shape

        rank = min(state.matrix_approximation_rank, rows, cols)

        original_main_elems = rows * cols
        compressed_elems = rank * (rows + cols)

        use_compression = (
            state.should_compress()
            and rank > 0
            and compressed_elems * state.min_compression_rate < original_main_elems
        )

        if use_compression:
            matrix_fp32 = matrix.float()

            q = _make_shared_random_q(
                rows=cols,
                rank=rank,
                device=device,
                state=state,
                hook_id=hook_id,
            )

            p = matrix_fp32.matmul(q)

            dist.all_reduce(
                p,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            p.div_(state.world_size)

            p = _orthogonalize(p)

            q = matrix_fp32.t().matmul(p)

            dist.all_reduce(
                q,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            q.div_(state.world_size)

            approx = p.matmul(q.t())

            reconstructed[:main_numel].copy_(
                approx.reshape(-1).to(dtype=original_dtype)
            )

            compressed_bytes += (p.numel() + q.numel()) * elem_size

        else:
            tmp = matrix.contiguous()
            dist.all_reduce(
                tmp,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            tmp.div_(state.world_size)

            reconstructed[:main_numel].copy_(tmp.reshape(-1))
            compressed_bytes += original_main_elems * elem_size

    if tail_numel > 0:
        tail = tensor[main_numel:].contiguous()

        dist.all_reduce(
            tail,
            op=dist.ReduceOp.SUM,
            group=state.process_group,
        )
        tail.div_(state.world_size)

        reconstructed[main_numel:].copy_(tail)
        compressed_bytes += tail_numel * elem_size

    return reconstructed, compressed_bytes


def fsdp_powersgd_error_compensated_x_hook(
    state: FSDPPowerSGDErrorCompensatedXState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    op_name = (
        "powersgd_error_compensated_x_reduce_scatter"
        if output is not None
        else "powersgd_error_compensated_x_all_reduce"
    )

    original_bytes = grad.numel() * grad.element_size()

    _sync_cuda_if_needed(grad)
    t0 = time.perf_counter()

    if not state.should_compress():
        _sync_cuda_if_needed(grad)
        t0 = time.perf_counter()

        if output is None:
            dist.all_reduce(
                grad,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            grad.div_(state.world_size)
        else:
            dist.reduce_scatter_tensor(
                output,
                grad,
                op=dist.ReduceOp.SUM,
                group=state.process_group,
            )
            output.div_(state.world_size)

        _sync_cuda_if_needed(grad)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        state.add_record(
            op_name="warmup_no_compression",
            original_bytes=original_bytes,
            compressed_bytes=original_bytes,
            latency_ms=latency_ms,
        )
        return

    hook_id, delta_prev, delta_prev2, error_memory = state.get_error_state(grad)
    
    alpha_t = state.get_alpha_t()
    alpha_prev = state.get_alpha_prev()
    alpha_prev2 = state.get_alpha_prev2()

    alpha_t = max(alpha_t, state.eps)

    coef_prev = alpha_prev / alpha_t * (2.0 - alpha_t)
    coef_prev2 = alpha_prev2 / alpha_t * (1.0 - alpha_t)

    new_error_memory = error_memory.detach().clone()
    new_error_memory.mul_(1.0 - state.beta)
    new_error_memory.add_(delta_prev, alpha=state.beta * coef_prev)
    new_error_memory.add_(delta_prev2, alpha=-state.beta * coef_prev2)

    ### == DEBUG ===
    grad_norm = torch.norm(grad.float())
    mem_norm = torch.norm(new_error_memory.float())
    max_mem_norm = 0.1 * grad_norm

    if mem_norm > max_mem_norm:
        new_error_memory.mul_(max_mem_norm / (mem_norm + state.eps))
    ### == DEBUG ===
    
    corrected_grad = grad.detach().clone()
    corrected_grad.add_(new_error_memory)

    reconstructed, compressed_bytes = _powersgd_compress_reconstruct(
        corrected_grad,
        state,
        hook_id,
    )

    delta_t = corrected_grad - reconstructed

    state.save_error_state(
        hook_id=hook_id,
        delta_t=delta_t,
        delta_prev=delta_prev,
        error_memory=new_error_memory,
    )

    state.add_residual_record(
        hook_id=hook_id,
        delta_prev_norm=torch.norm(delta_prev.float()).detach().item(),
        delta_prev2_norm=torch.norm(delta_prev2.float()).detach().item(),
        memory_norm=torch.norm(new_error_memory.float()).detach().item(),
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

    _sync_cuda_if_needed(grad)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    state.add_record(
        op_name=op_name,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        latency_ms=latency_ms,
    )

    # DEBUG
    # grad_norm = torch.norm(grad.float()).item()
    # memory_norm = torch.norm(new_error_memory.float()).item()
    # delta_norm = torch.norm(delta_t.float()).item()
    # print(grad_norm, memory_norm, delta_norm)

def register_fsdp_powersgd_error_compensated_x_hook(
    model,
    rank,
    world_size,
    matrix_approximation_rank=2,
    block_cols=4096,
    min_compression_rate=2.0,
    alpha=0.9,
    beta=1.0,
    start_compression_iter=1,
    seed=1234,
):
    state = FSDPPowerSGDErrorCompensatedXState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
        matrix_approximation_rank=matrix_approximation_rank,
        block_cols=block_cols,
        min_compression_rate=min_compression_rate,
        alpha=alpha,
        beta=beta,
        start_compression_iter=start_compression_iter,
        seed=seed,
    )

    model.register_comm_hook(
        state,
        fsdp_powersgd_error_compensated_x_hook,
    )

    return state