#! /bin/bash

# train
CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_ds/ -O --iters 200000 --asr_model deepspeech
CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_ds/ -O --finetune_lips --iters 250000 --asr_model deepspeech

CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_ds_torso/ -O --torso --iters 200000 --head_ckpt trial_obama_ds/checkpoints/ngp_ep0035.pth --asr_model deepspeech

# test
CUDA_VISIBLE_DEVICES=1 python main.py data/obama/ --workspace trial_obama_ds_torso/ -O --torso --test --asr_model deepspeech