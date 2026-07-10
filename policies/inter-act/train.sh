#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./inter_act_ckpt/inter-act-${task_name}/${task_config}-${expert_data_num} \
    --policy_class InterACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_blocks 3 \
    --num_cls_tokens_arm 3 \
    --num_cls_tokens_image 3 \
    --n_pre_decoder_layers 2 \
    --n_post_decoder_layers 2 \
    --n_sync_decoder_layers 1 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 16 \
    --seed ${seed}
