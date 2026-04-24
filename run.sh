torchrun \
  --rdzv-id=exp001 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR \
  --nnodes=2 \
  --nproc-per-node=1 \
  train.py
  #> train_$(hostname).log 2>&1 &