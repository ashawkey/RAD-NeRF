#! /bin/bash

# train
CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_eo/ -O --iters 200000
CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_eo/ -O --finetune_lips --iters 250000

CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_eo_torso/ -O --torso --iters 200000 --head_ckpt trial_obama_eo/checkpoints/ngp_ep0035.pth

# test
CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_eo_torso/ -O --torso --test