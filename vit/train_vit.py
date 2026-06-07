import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import os
import json
import time
import math
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import timm
from timm.models.vision_transformer import Block as ViTBlock

from methods.int8 import register_fsdp_int8_comm_hook
from methods.int8_errcomp import register_fsdp_int8_error_feedback_hook
from methods.topk import register_fsdp_sparse_topk_hook
from methods.topk_errcomp import register_fsdp_sparse_topk_error_feedback_hook
from methods.topk_adaptive import register_fsdp_sparse_topk_adaptive_error_feedback_hook
from methods.int8_adaptive import register_fsdp_int8_adaptive_error_feedback_hook
from methods.int8_direction import register_fsdp_int8_direction_aware_error_feedback_hook
from methods.int8_clipped import register_fsdp_int8_block_clipped_error_feedback_hook
from methods.int8_clipped_trust_dual import register_fsdp_int8_block_clipped_trust_dual_error_feedback_hook
from methods.int8_bias_correction import register_fsdp_int8_bias_correction_hook
from methods.int8_linear_calibration import register_fsdp_int8_linear_calibration_hook
from methods.signsgd_ef import register_fsdp_scaled_sign_ef_hook
from methods.fsdp_powersgd import register_fsdp_powersgd_hook
from methods.fsdp_powersgd_clipped_trust_dual import (
    register_fsdp_powersgd_clipped_trust_dual_ef_hook,
)
from methods.fsdp_powersgd_direction import (
    register_fsdp_powersgd_direction_aware_error_feedback_hook,
)
from methods.fsdp_powersgd_norm_gated_ef import (
    register_fsdp_powersgd_norm_gated_error_feedback_hook,
)
from methods.fsdp_powersgd_plus import (
    register_fsdp_powersgd_plus_error_feedback_hook,
)
from methods.fsdp_powersgd_two_step_ef import (
    register_fsdp_powersgd_two_step_error_feedback_hook,
)
from methods.int8_liec import register_fsdp_int8_liec_hook
from methods.fsdp_powersgd_x import register_fsdp_powersgd_error_compensated_x_hook

# -----------------
# Config
# -----------------

DATA_DIR = os.path.dirname(os.path.dirname(__file__)) + "/data/cifar10/data"
NUM_CLASSES = 10
IMAGE_SIZE = 224

BATCH_SIZE = 128          # per GPU
NUM_WORKERS = 4
MAX_ITERS = 20000

LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 500
LR_DECAY_ITERS = MAX_ITERS
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.999

LOG_INTERVAL = 10
EVAL_INTERVAL = 500
EVAL_ITERS = 50

DTYPE = "float32"       # float32 | bfloat16 | float16
BACKEND = "nccl"
GRAD_CLIP = 1.0


# -----------------
# Communication logging
# -----------------

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
class FSDPCommsLoggerState:
    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    summary: dict = field(default_factory=dict)

    def reset(self):
        self.summary = {}

    def add_record(self, op_name, msg_size_bytes, latency_ms):
        key = str(msg_size_bytes)
        self.summary.setdefault(op_name, {})
        self.summary[op_name].setdefault(key, {
            "count": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "tput_avg_gbps": 0.0,
            "busbw_avg_gbps": 0.0,
            "msg_size_bytes": msg_size_bytes,
            "msg_size_str": format_bytes(msg_size_bytes),
        })

        rec = self.summary[op_name][key]
        rec["count"] += 1
        rec["total_latency_ms"] += latency_ms
        rec["avg_latency_ms"] = rec["total_latency_ms"] / rec["count"]

        if rec["avg_latency_ms"] > 0:
            rec["tput_avg_gbps"] = (
                msg_size_bytes / (rec["avg_latency_ms"] / 1000.0)
            ) / 1e9
            rec["busbw_avg_gbps"] = rec["tput_avg_gbps"]


def fsdp_comms_logging_hook(
    state: FSDPCommsLoggerState,
    grad: torch.Tensor,
    output: torch.Tensor | None = None,
):
    if output is None:
        op_name = "all_reduce"
        msg_size_bytes = grad.numel() * grad.element_size()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_reduce(grad, group=state.process_group)
        grad.div_(state.world_size)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        state.add_record(op_name, msg_size_bytes, latency_ms)
        return

    op_name = "reduce_scatter"
    msg_size_bytes = grad.numel() * grad.element_size()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    dist.reduce_scatter_tensor(
        output,
        grad,
        op=dist.ReduceOp.SUM,
        group=state.process_group,
    )
    output.div_(state.world_size)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    state.add_record(op_name, msg_size_bytes, latency_ms)


def register_fsdp_comms_logger(model, rank, world_size):
    state = FSDPCommsLoggerState(
        process_group=dist.group.WORLD,
        rank=rank,
        world_size=world_size,
    )
    model.register_comm_hook(state, fsdp_comms_logging_hook)
    return state


def build_comm_log_object(comm_summary, rank, world_size):
    return {
        "summary": comm_summary,
        "straggler_analysis": None,
        "metadata": {
            "world_size": world_size,
            "rank": rank,
            "timestamp": datetime.now().isoformat(),
        },
    }


# -----------------
# Utils
# -----------------

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def setup_distributed():
    init_process_group(backend=BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"world_size={world_size}", flush=True)

    return rank, world_size, local_rank, device


def cleanup_distributed():
    destroy_process_group()


def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR

    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def cycle(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# -----------------
# Data
# -----------------

def build_dataloaders(rank, world_size):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=(rank == 0),
        transform=train_transform,
    )
    dist.barrier()

    val_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=(rank == 0),
        transform=val_transform,
    )
    dist.barrier()

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, train_sampler, val_loader


# -----------------
# Model
# -----------------

def vit_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if recurse:
        return True
    return isinstance(module, ViTBlock)


def build_model(device):
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=vit_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )

    return model, optimizer


# -----------------
# Eval
# -----------------

@torch.no_grad()
def estimate_metrics(model, val_loader, device, ctx):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_count = torch.tensor(0, device=device, dtype=torch.long)

    for i, (x, y) in enumerate(val_loader):
        if i >= EVAL_ITERS:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with ctx:
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.detach() * x.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum()
        total_count += x.size(0)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    metrics = {
        "val_loss": (total_loss / total_count).item(),
        "val_acc": (total_correct.float() / total_count.float()).item(),
    }

    model.train()
    return metrics


# -----------------
# Communication hooks
# -----------------

def register_comm_hook(args, model, rank, world_size):
    if args.comm_mode == "none":
        return None
    if args.comm_mode == "log":
        return register_fsdp_comms_logger(model, rank, world_size)
    if args.comm_mode == "int8":
        return register_fsdp_int8_comm_hook(model, rank, world_size)
    if args.comm_mode == "int8_ef":
        return register_fsdp_int8_error_feedback_hook(model, rank, world_size)
    if args.comm_mode == "sparse_topk":
        return register_fsdp_sparse_topk_hook(model, rank, world_size, args.sparsity)
    if args.comm_mode == "sparse_topk_ef":
        return register_fsdp_sparse_topk_error_feedback_hook(model, rank, world_size, args.sparsity)
    if args.comm_mode == "int8_clipped_trust_dual_ef":
        return register_fsdp_int8_block_clipped_trust_dual_error_feedback_hook(
            model=model,
            rank=rank,
            world_size=world_size,
            block_size=4096,
            percentile=0.999,
            mu=0.95,
            rho=0.9,
            lambda_long=0.5,
        )
    if args.comm_mode == "powersgd":
        return register_fsdp_powersgd_hook(
            model=model,
            rank=rank,
            world_size=world_size,
            matrix_approximation_rank=args.powersgd_rank,
            block_cols=args.powersgd_block_cols,
            min_compression_rate=args.powersgd_min_compression_rate,
            use_error_feedback=False,
            start_compression_iter=args.powersgd_start_iter,
        )
    if args.comm_mode == "powersgd_ef":
        return register_fsdp_powersgd_hook(
            model=model,
            rank=rank,
            world_size=world_size,
            matrix_approximation_rank=args.powersgd_rank,
            block_cols=args.powersgd_block_cols,
            min_compression_rate=args.powersgd_min_compression_rate,
            use_error_feedback=True,
            start_compression_iter=args.powersgd_start_iter,
        )

    if args.comm_mode == "powersgd_norm_gated_ef":
        return register_fsdp_powersgd_norm_gated_error_feedback_hook(
            model=model,
            rank=rank,
            world_size=world_size,
            matrix_approximation_rank=args.powersgd_rank,
            block_cols=args.powersgd_block_cols,
            min_compression_rate=args.powersgd_min_compression_rate,
            tau=args.powersgd_tau,
            residual_momentum=args.powersgd_residual_momentum,
            start_compression_iter=args.powersgd_start_iter,
        )
    if args.comm_mode == "x":
        return register_fsdp_powersgd_error_compensated_x_hook(
            model=model,
            rank=rank,
            world_size=world_size,
            matrix_approximation_rank=args.powersgd_rank,
            block_cols=args.powersgd_block_cols,
            min_compression_rate=args.powersgd_min_compression_rate,
            alpha=0.3,#0.9,
            beta=0.3,#1.0,
            start_compression_iter=args.powersgd_start_iter,
        )
    if args.comm_mode == "powersgd_plus_ef":
        return register_fsdp_powersgd_plus_error_feedback_hook(
            model=model,
            rank=rank,
            world_size=world_size,
            matrix_approximation_rank=args.powersgd_rank,
            block_cols=args.powersgd_block_cols,
            min_compression_rate=args.powersgd_min_compression_rate,
            svd_refresh_period=args.powersgd_svd_refresh_period,
            start_compression_iter=args.powersgd_start_iter,
            use_error_feedback=True,
        )
    raise ValueError(f"Unknown comm_mode: {args.comm_mode}")


# -----------------
# Train
# -----------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs_vit_tiny_fsdp")
    parser.add_argument(
        "--comm_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "log",
            "int8",
            "int8_ef",
            "sparse_topk",
            "sparse_topk_ef",
            "sparse_topk_adaptive_ef",
            "int8_adaptive_ef",
            "int8_direction_ef",
            "int8_clipped_ef",
            "int8_clipped_trust_dual_ef",
            "int8_bias_correction",
            "int8_linear_calibration",
            "signsgd_ef",
            "powersgd",
            "powersgd_ef",
            "powersgd_clipped_trust_dual_ef",
            "powersgd_direction_ef",
            "powersgd_norm_gated_ef",
            "powersgd_plus_ef",
            "powersgd_two_step_ef",
            "int8_liec",
            "x"
        ],
    )
    parser.add_argument("--sparsity", type=float, default=0.99)
    parser.add_argument("--powersgd_rank", type=int, default=1)
    parser.add_argument("--powersgd_block_cols", type=int, default=4096)
    parser.add_argument("--powersgd_min_compression_rate", type=float, default=2.0)
    parser.add_argument("--powersgd_tau", type=float, default=0.25)
    parser.add_argument("--powersgd_residual_momentum", type=float, default=0.9)
    # powersgd+
    parser.add_argument("--powersgd_svd_refresh_period", type=int, default=50)
    parser.add_argument("--powersgd_start_iter", type=int, default=10)
    # powersgd two steps
    parser.add_argument("--powersgd_gamma_prev", type=float, default=0.5)
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    is_master = rank == 0

    if is_master:
        os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, "loss_vit_tiny_fsdp.jsonl")
    if is_master and os.path.exists(log_path):
        os.remove(log_path)
    dist.barrier()

    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[DTYPE]

    ctx = nullcontext() if DTYPE == "float32" else torch.amp.autocast(
        device_type="cuda",
        dtype=ptdtype,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == "float16"))

    train_loader, train_sampler, val_loader = build_dataloaders(rank, world_size)
    train_iter = cycle(train_loader, train_sampler)

    model, optimizer = build_model(device)
    criterion = nn.CrossEntropyLoss()
    comm_state = register_comm_hook(args, model, rank, world_size)

    if is_master:
        print(f"model=vit_tiny_patch16_224, params={sum(p.numel() for p in model.parameters())}", flush=True)
        print(f"batch_size_per_rank={BATCH_SIZE}, global_batch_size={BATCH_SIZE * world_size}", flush=True)

    start_time = time.time()

    print(
        f"[rank={rank}] local_rank={local_rank} "
        f"device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )

    for iter_num in range(MAX_ITERS):
        if comm_state is not None:
            comm_state.reset()

        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with ctx:
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if GRAD_CLIP != 0.0:
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        do_log = iter_num % LOG_INTERVAL == 0
        do_eval = iter_num % EVAL_INTERVAL == 0

        metrics = None
        if do_eval:
            metrics = estimate_metrics(model, val_loader, device, ctx)

        if do_log or do_eval:
            comm_obj = None

            if comm_state is not None:
                local_comm_obj = build_comm_log_object(
                    comm_state.summary,
                    rank,
                    world_size,
                )
                gathered_comm = [None for _ in range(world_size)] if rank == 0 else None
                dist.gather_object(
                    obj=local_comm_obj,
                    object_gather_list=gathered_comm,
                    dst=0,
                )

                if rank == 0:
                    comm_obj = {
                        "summary": gathered_comm[0]["summary"],
                        "straggler_analysis": None,
                        "metadata": {
                            "world_size": world_size,
                            "rank": 0,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "by_rank": gathered_comm,
                    }

            if is_master:
                row = {
                    "iter": iter_num,
                    "loss": loss.detach().float().item(),
                    "lr": lr,
                    "elapsed_sec": time.time() - start_time,
                    "world_size": world_size,
                }
                if metrics is not None:
                    row.update(metrics)
                if comm_state is not None:
                    row["comm"] = comm_obj

                append_jsonl(log_path, row)
                print(row, flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    train()
