#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 \
python -m \
torch.distributed.launch \
--nproc_per_node=4 \
main.py \
--cfg ./config/train.yaml \
# --world-size 2 \
