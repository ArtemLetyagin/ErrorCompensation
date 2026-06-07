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

from model import GPTConfig, GPT


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

GRADIENT_ACCUMULATION_STEPS = 40

LOG_INTERVAL = 10

DTYPE = "float32"
BACKEND = "nccl"
GRAD_CLIP = 0.0


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

def iter_trainable_params(model):
    for p in model.parameters():
        if p.requires_grad:
            yield p


def get_or_init_error(error_dict, name, tensor):
    err = error_dict.get(name)

    if (
        err is None
        or err.shape != tensor.shape
        or err.device != tensor.device
        or err.dtype != tensor.dtype
    ):
        err = torch.zeros_like(tensor)
        error_dict[name] = err

    return err

class LIECLogger:
    def __init__(self):
        self.summary = {}

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
                "msg_size_bytes": msg_size_bytes,
                "msg_size_str": format_bytes(msg_size_bytes),
            }

        rec = self.summary[op_name][key]
        rec["count"] += 1
        rec["total_latency_ms"] += latency_ms
        rec["avg_latency_ms"] = rec["total_latency_ms"] / rec["count"]


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

    return model, optimizer


def flatten_params(model):
    return torch.cat([
        p.data.view(-1)
        for p in model.parameters()
        if p.requires_grad
    ])


def set_flat_params(model, flat):
    offset = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue

        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p.data))
        offset += numel


def flatten_grads(model):
    grads = []

    for p in model.parameters():
        if not p.requires_grad:
            continue

        if p.grad is None:
            grads.append(torch.zeros_like(p.data).view(-1))
        else:
            grads.append(p.grad.data.view(-1))

    return torch.cat(grads)


def assign_flat_grads(model, flat_grad):
    offset = 0

    for p in model.parameters():
        if not p.requires_grad:
            continue

        numel = p.numel()

        if p.grad is None:
            p.grad = torch.zeros_like(p.data)

        p.grad.data.copy_(flat_grad[offset:offset + numel].view_as(p.data))
        offset += numel


def sign_compress(x, eps=1e-8):
    scale = x.abs().mean().clamp_min(eps)

    signs = torch.where(
        x >= 0,
        torch.ones_like(x, dtype=torch.int8),
        -torch.ones_like(x, dtype=torch.int8),
    )

    return signs, scale.to(torch.float32)

def sign_compress_fp(x, eps=1e-8):
    scale = x.abs().mean().clamp_min(eps)

    signs = torch.empty_like(x)
    signs.copy_(x)
    signs.sign_()
    signs.masked_fill_(signs == 0, 1.0)

    return signs, scale

def sign_decompress_to(signs, scale, out):
    out.copy_(signs)
    out.mul_(scale.to(dtype=out.dtype))
    return out


@torch.no_grad()
def liec_bidirectional_compress_paramwise(
    model,
    global_errors,
    rank,
    world_size,
    logger,
    eps=1e-8,
):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_msg_bytes = 0

    for param_idx, p in enumerate(iter_trainable_params(model)):
        if p.grad is None:
            continue

        grad = p.grad.data

        global_error = get_or_init_error(
            global_errors,
            param_idx,
            grad,
        )

        # -------------------------
        # 1. Worker-side compression
        # -------------------------
        signs_i, scale_i = sign_compress_fp(grad, eps=eps)

        p_i = torch.empty_like(grad)
        sign_decompress_to(signs_i, scale_i, p_i)

        # local_error = grad - p_i
        local_error = torch.empty_like(grad)
        local_error.copy_(grad)
        local_error.sub_(p_i)

        # -------------------------
        # 2. Gather compressed local gradients
        # -------------------------
        signs_list = [torch.empty_like(signs_i) for _ in range(world_size)]
        scales_list = [torch.empty_like(scale_i) for _ in range(world_size)]

        dist.all_gather(signs_list, signs_i)
        dist.all_gather(scales_list, scale_i)

        avg_p = torch.zeros_like(grad)

        tmp = torch.empty_like(grad)

        for signs_j, scale_j in zip(signs_list, scales_list):
            sign_decompress_to(signs_j, scale_j, tmp)
            avg_p.add_(tmp)

        avg_p.div_(world_size)

        # -------------------------
        # 3. Server/global compression
        # -------------------------
        avg_p.add_(global_error)

        signs_g, scale_g = sign_compress_fp(avg_p, eps=eps)

        p_global = torch.empty_like(grad)
        sign_decompress_to(signs_g, scale_g, p_global)

        # global_error = corrected - p_global
        global_error.copy_(avg_p)
        global_error.sub_(p_global)

        # -------------------------
        # 4. Immediate local compensation
        # p.grad = p_global + local_error
        # -------------------------
        grad.copy_(p_global)
        grad.add_(local_error)

        total_msg_bytes += (
            world_size * (signs_i.numel() * 1 + 4)
            + world_size * (signs_g.numel() * 1 + 4)
        )

        del signs_i, p_i, local_error
        del signs_list, scales_list
        del avg_p, tmp, signs_g, p_global

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logger.add_record(
        op_name="liec_paramwise_bidirectional_sign",
        msg_size_bytes=total_msg_bytes,
        latency_ms=latency_ms,
    )


@torch.no_grad()
def average_model_parameters(model, logger, world_size):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    flat = flatten_params(model)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    set_flat_params(model, flat)

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logger.add_record(
        op_name="model_average_all_reduce",
        msg_size_bytes=flat.numel() * flat.element_size(),
        latency_ms=latency_ms,
    )


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs_liec")
    parser.add_argument("--average_period", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    is_master = rank == 0

    if is_master:
        os.makedirs(args.log_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, "loss_liec.jsonl")

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

    assert GRADIENT_ACCUMULATION_STEPS % world_size == 0
    grad_accum_steps = GRADIENT_ACCUMULATION_STEPS // world_size

    global_error = torch.zeros_like(flatten_grads(model))
    logger = LIECLogger()

    if is_master:
        print(f"world_size={world_size}", flush=True)
        print(f"grad_accum_steps_per_rank={grad_accum_steps}", flush=True)
        print(f"average_period={args.average_period}", flush=True)

    start_time = time.time()

    for iter_num in range(MAX_ITERS):
        print(iter_num, flush=True)

        logger.reset()

        lr = get_lr(iter_num)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = dataset.get_batch("train", device)

            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum_steps

            total_loss += loss.detach().float().item()
            scaler.scale(loss).backward()

        if GRAD_CLIP != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        flat_grad = flatten_grads(model)

        if (iter_num + 1) % args.average_period == 0:
            average_model_parameters(model, logger, world_size)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            for p in iter_trainable_params(model):
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data.div_(world_size)

            global_errors = {}

            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            total_grad_bytes = sum(
                p.numel() * p.element_size()
                for p in iter_trainable_params(model)
                if p.grad is not None
            )

            logger.add_record(
                op_name="exact_paramwise_grad_average_all_reduce",
                msg_size_bytes=total_grad_bytes,
                latency_ms=latency_ms,
            )

        else:
            liec_bidirectional_compress_paramwise(
                model=model,
                global_errors=global_errors,
                rank=rank,
                world_size=world_size,
                logger=logger,
                eps=args.eps,
            )

        assign_flat_grads(model, update_grad)

        scaler.step(optimizer)
        scaler.update()

        if iter_num % LOG_INTERVAL == 0:
            if is_master:
                row = {
                    "iter": iter_num,
                    "loss": total_loss,
                    "lr": lr,
                    "elapsed_sec": time.time() - start_time,
                    "world_size": world_size,
                    "grad_accum_steps": grad_accum_steps,
                    "effective_batch_size": BATCH_SIZE * grad_accum_steps * world_size,
                    "average_period": args.average_period,
                    "comm": {
                        "summary": logger.summary,
                    },
                }

                append_jsonl(log_path, row)
                print(row, flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    train()