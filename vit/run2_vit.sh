export MASTER_ADDR=192.168.3.130
export MASTER_PORT=29501
# export NCCL_SOCKET_IFNAME=enp5s0f0np0
# export NCCL_SOCKET_IFNAME=eno2 # работало до работ
export NCCL_SOCKET_IFNAME=enp5s0f0np0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
unset TORCH_DISTRIBUTED_DEBUG

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_vit.py \
  --comm_mode x \
  --sparsity 0.99 \
  --powersgd_rank 16 \
  --powersgd_block_cols 2048 \
  --powersgd_min_compression_rate 2.0 \
  --powersgd_tau 0.25 \
  --powersgd_residual_momentum 0.9 \
  --powersgd_svd_refresh_period 5 \
  --powersgd_start_iter 0 \
  --powersgd_gamma_prev 0.5