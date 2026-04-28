import os
import json
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPTConfig, GPT

from dataclasses import dataclass, field
from datetime import datetime
from int8 import register_fsdp_int8_comm_hook
from int8_errcomp import register_fsdp_int8_error_feedback_hook
from methods.topk import register_fsdp_sparse_topk_hook
from methods.topk_errcomp import register_fsdp_sparse_topk_error_feedback_hook

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
            # GB/s-like DeepSpeed style: bytes / sec / 1e9
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
# Config
# -----------------

DATASET = "shakespeare"
DATA_DIR = os.path.join("data", DATASET)

BATCH_SIZE = 6
BLOCK_SIZE = 512

N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
DROPOUT = 0.0
BIAS = False
VOCAB_SIZE = 50304

LEARNING_RATE = 6e-4
MIN_LR = 6e-5
MAX_ITERS = 600000
WARMUP_ITERS = 2000
LR_DECAY_ITERS = MAX_ITERS
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95

# Global accumulation, same logic as nanoGPT/DDP:
# per-rank accumulation = GRADIENT_ACCUMULATION_STEPS // world_size
GRADIENT_ACCUMULATION_STEPS = 40

LOG_INTERVAL = 10
EVAL_INTERVAL = 1000
EVAL_ITERS = 5

DTYPE = "float32"
BACKEND = "nccl"

GRAD_CLIP = 0.0
COMPRESS_ALL_LAYERS = False


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


# -----------------
# Data
# -----------------

class BinaryDataset:
    def __init__(self, data_dir):
        self.train_data = np.memmap(
            os.path.join(data_dir, "train.bin"),
            dtype=np.uint16,
            mode="r",
        )
        self.val_data = np.memmap(
            os.path.join(data_dir, "val.bin"),
            dtype=np.uint16,
            mode="r",
        )

    def get_batch(self, split, device):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

        x = torch.stack([
            torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64))
            for i in ix
        ])

        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64))
            for i in ix
        ])

        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)

        return x, y


# -----------------
# Model
# -----------------

def custom_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if not recurse:
        return isinstance(module, torch.nn.Embedding) or isinstance(module, torch.nn.Linear)

    threshold = 100000 if COMPRESS_ALL_LAYERS else 2000000
    return nonwrapped_numel > threshold


def build_model(device):
    model_args = dict(
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        bias=BIAS,
        vocab_size=VOCAB_SIZE,
        dropout=DROPOUT,
    )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(
        WEIGHT_DECAY,
        LEARNING_RATE,
        (BETA1, BETA2),
        "cuda",
    )

    model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=custom_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )

    # model = DDP(model, device_ids=[0])

    

    return model, optimizer


# -----------------
# Eval
# -----------------

@torch.no_grad()
def estimate_loss(model, dataset, device, ctx):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS, device=device)

        for k in range(EVAL_ITERS):
            x, y = dataset.get_batch(split, device)

            with ctx:
                _, loss = model(x, y)

            losses[k] = loss.detach()

        local_mean = losses.mean()

        dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
        global_mean = local_mean / dist.get_world_size()

        out[split] = global_mean.item()

    model.train()
    return out


# -----------------
# Train
# -----------------

def train():
    # === ARGS ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs_fsdp")

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
        ],
    )

    parser.add_argument("--sparsity", type=float, default=0.99)

    args = parser.parse_args()
    # === ARGS ===

    rank, world_size, local_rank, device = setup_distributed()
    is_master = rank == 0

    if is_master:
        os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, "loss_fsdp.jsonl")

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

    dataset = BinaryDataset(DATA_DIR)
    model, optimizer = build_model(device)
    comm_state = None

    # === РЕГИСТРАЦИЯ ХУКА ===
    comm_state = None

    if args.comm_mode == "log":
        comm_state = register_fsdp_comms_logger(model, rank, world_size)

    elif args.comm_mode == "int8":
        comm_state = register_fsdp_int8_comm_hook(model, rank, world_size)

    elif args.comm_mode == "int8_ef":
        comm_state = register_fsdp_int8_error_feedback_hook(model, rank, world_size)

    elif args.comm_mode == "sparse_topk":
        comm_state = register_fsdp_sparse_topk_hook(model, rank, world_size, args.sparsity)
    
    elif args.comm_mode == "sparse_topk_ef":
        comm_state = register_fsdp_sparse_topk_error_feedback_hook(model, rank, world_size, args.sparsity)
    # === РЕГИСТРАЦИЯ ХУКА ===

    assert GRADIENT_ACCUMULATION_STEPS % world_size == 0
    grad_accum_steps = GRADIENT_ACCUMULATION_STEPS // world_size

    if is_master:
        print(f"grad_accum_steps_per_rank={grad_accum_steps}", flush=True)

    start_time = time.time()

    print(
        f"[rank={rank}] local_rank={local_rank} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"current_device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )

    for iter_num in range(MAX_ITERS):
        print(iter_num, flush=True)
        if comm_state is not None:
            comm_state.reset()

        lr = get_lr(iter_num)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = dataset.get_batch("train", device)

            if micro_step != grad_accum_steps - 1:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                with ctx:
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps

                total_loss += loss.detach().float().item()
                scaler.scale(loss).backward()

        if GRAD_CLIP != 0.0:
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        # =====
        if iter_num % LOG_INTERVAL == 0:
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
                    "loss": total_loss,
                    "lr": lr,
                    "elapsed_sec": time.time() - start_time,
                    "world_size": world_size,
                }

                if comm_state is not None:
                    row["comm"] = comm_obj

                append_jsonl(log_path, row)
                print(row, flush=True)
        # =====


    cleanup_distributed()


if __name__ == "__main__":
    train()