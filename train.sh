#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5,6,7 \
python -m \
torch.distributed.launch \
--nproc_per_node=3 \
main.py \
--cfg ./config/train.yaml \
# --world-size 2 \
