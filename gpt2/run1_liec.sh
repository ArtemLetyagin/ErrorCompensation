export MASTER_ADDR=192.168.3.130
export MASTER_PORT=29501
# export NCCL_SOCKET_IFNAME=enp4s0f0np0
# export NCCL_SOCKET_IFNAME=enp6s0f1 # работало до работ
export NCCL_SOCKET_IFNAME=enp4s0f0np0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
unset TORCH_DISTRIBUTED_DEBUG

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_liec.py \
  --average_period 32 \
  --log_dir logs_liec