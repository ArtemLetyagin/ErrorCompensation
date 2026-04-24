import numpy as np
import os
import torch
from model import GPTConfig, GPT
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from contextlib import nullcontext
import time
# from tagc import transformer_compress_hook, IndexSize, TAGCState

# +-----------+
# | CONSTANTS |
# +-----------+

DATASET = 'shakespeare'
DATA_DIR = os.path.join('data', DATASET)
BATCH_SIZE = 6#12
BLOCK_SIZE = 512#1024
DEVICE = 'cuda'
DEVICE_TYPE = 'cuda'

# --- model ---
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
DROPOUT = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
BIAS = False # do we use bias inside LayerNorm and Linear layers?

META_VOCAB_SIZE = None
COMPRESS_ALL_LAYERS = False

# --- ddp ---
BACKEND = 'gloo' # 'nccl', 'gloo', etc.
GRADIENT_ACCUMULATION_STEPS = 5 * 8

# --- Adam ---
LEARNING_RATE = 6e-4 # max learning rate
MAX_ITERS = 600000 # total number of training iterations
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95
DECAY_LR = True
WARMUP_ITERS = 2000

EVAL_ITERS = 10 # 200

dtype = 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if DEVICE_TYPE == 'cpu' else torch.amp.autocast(device_type=DEVICE_TYPE, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# +-----+
# | DDP |
# +-----+

ddp = True # is this a ddp run?
if ddp:
    init_process_group(backend=BACKEND)

    ddp_rank = dist.get_rank()
    ddp_world_size = dist.get_world_size()
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(ddp_rank, ddp_world_size, ddp_local_rank)
    # ddp_rank = int(os.environ['RANK'])
    # ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # ddp_world_size = int(os.environ['WORLD_SIZE'])

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert GRADIENT_ACCUMULATION_STEPS % ddp_world_size == 0
    GRADIENT_ACCUMULATION_STEPS //= ddp_world_size

# +------+
# | DATA |
# +------+

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    if DEVICE_TYPE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# +-------+
# | MODEL |
# +-------+

def custom_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if not recurse:
        return isinstance(module, torch.nn.Embedding) or isinstance(module, torch.nn.Linear)
    return nonwrapped_numel > (100000 if COMPRESS_ALL_LAYERS else 2000000)

# model init
def get_model():
    model_args = dict(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, block_size=BLOCK_SIZE,
                    bias=BIAS, vocab_size=None, dropout=DROPOUT)

    model_args['vocab_size'] = META_VOCAB_SIZE if META_VOCAB_SIZE is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    

    if BLOCK_SIZE < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = BLOCK_SIZE # so that the checkpoint will have the right value
    
    model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=custom_wrap_policy
    )

    model.to(device)
    
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), DEVICE_TYPE)
    return model, optimizer

def model_registre_hook(model):
    import os
    import socket
    import torch
    import torch.distributed as dist
    from dataclasses import dataclass

    @dataclass
    class DebugCommState:
        process_group: dist.ProcessGroup
        rank: int
        world_size: int
        max_print_elems: int = 8


    def debug_reduce_hook(state: DebugCommState, grad: torch.Tensor, output: torch.Tensor | None = None):
        """
        FSDP comm hook:
        - grad: full flattened unsharded gradient for this FSDP unit
        - output: preallocated shard buffer in sharded case
        """
        host = socket.gethostname()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Логируем не весь тензор, а метаданные + маленький сэмпл
        # sample = grad[: min(state.max_print_elems, grad.numel())].detach().cpu().tolist()
        # print(
        #     f"[host={host} rank={state.rank} local_rank={local_rank}] "
        #     f"HOOK grad.numel={grad.numel()} grad.dtype={grad.dtype} "
        #     f"grad.device={grad.device} grad.norm={grad.norm().item():.6f} "
        #     ,#f"sample={sample}",
        #     flush=True,
        # )

        # NO_SHARD case: просто обычный all-reduce с усреднением
        if output is None:
            # print("ALL REDUCE")
            dist.all_reduce(grad, group=state.process_group)
            grad.div_(state.world_size)
            return

        # SHARDED case:
        # режем полный flattened grad на равные куски по числу процессов
        assert grad.ndim == 1, "Expected flattened 1D grad"
        shard_size = grad.numel() // state.world_size
        assert shard_size * state.world_size == grad.numel(), "For demo: require divisible grad size"

        for dst in range(state.world_size):
            shard = grad[dst * shard_size : (dst + 1) * shard_size]
            if dst == state.rank:
                # print("COPY REDUCE")
                output.copy_(shard)
                dist.reduce(output, dst=dst, group=state.process_group)
                output.div_(state.world_size)
            else:
                # print("REDUCE")
                dist.reduce(shard, dst=dst, group=state.process_group)

    from torch.distributed.distributed_c10d import _get_default_group

    state = DebugCommState(
        process_group=_get_default_group(),
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )

    model.register_comm_hook(state, debug_reduce_hook)
    # is_transformer_hook = (lambda numel: numel != 2359296 and numel != 38633472
    #                     and (not COMPRESS_ALL_LAYERS
    #                     or numel != 1769472 and numel != 589824 and numel != 393216))
    # state = TAGCState(process_group=_get_default_group(),
    #                     num_processes=ddp_world_size,
    #                     process_index=ddp_rank,
    #                     sparsify_fraction=0.9875,
    #                     index_size=IndexSize.FOUR_BITS,
    #                     compress_ratio=10,
    #                     device=device,
    #                     is_transformer_hook=is_transformer_hook,
    #                     parameter_type=torch.float32)
    # model.register_comm_hook(state, transformer_compress_hook)
    return model

# +------------+
# | VALIDATING |
# +------------+

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()

            #print(f"[RANK={ddp_rank}] LOSS={loss.item()}")
        out[split] = losses.mean()
    model.train()
    return out

# +----+
# | LR |
# +----+

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (lr_decay_iters - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (LEARNING_RATE - min_lr)

def trace_handler(prof):
    print('trace_handler')
    prof.export_chrome_trace(
        f"{out_dir}/trace_{ddp_local_rank}_{prof.step_num}.json"
    )

last_tick = None
def timer_tick(message: str='', detail=False):
    global last_tick
#    return # Turn off extra logging
    current_time = time.time_ns()
    if last_tick is not None:
        if not detail:
            print(message + f'[RANK={ddp_rank}] timer tick {(current_time - last_tick) / 1_000_000}')
    last_tick = time.time_ns()

def training_loop(model, optimizer):
    iter_num = 0

    # training loop
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=190, warmup=9, active=1, repeat=0),
    #    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
        on_trace_ready=trace_handler,
    ) as prof:
        X, Y = get_batch('train')
        
        while True:
            timer_tick('Before get_lr')
            losses = estimate_loss()
            timer_tick(str(losses))
            X, Y = get_batch('train')
            
            lr = get_lr(iter_num) if DECAY_LR else LEARNING_RATE

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # losses = estimate_loss()
            # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
                with ctx:
                    # timer_tick('Before model', detail=True)
                    logits, loss = model(X, Y)
                    # timer_tick('After model')
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                # timer_tick('after backward')
                prof.step()
                # timer_tick('After step')
            
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            timer_tick('END')
           

if __name__ == "__main__":
    model, optimizer = get_model()
    model = model_registre_hook(model)
    training_loop(model, optimizer)
