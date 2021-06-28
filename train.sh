#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 \
python -m \
torch.distributed.launch \
--nproc_per_node=1 \
main.py \
--cfg ./config/train.yaml \
