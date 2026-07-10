#!/bin/bash

policy_name=inter_act
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
# temporal_agg=${5} # use temporal_agg
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/inter-act/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir policy/inter-act/inter_act_ckpt/inter-act-${task_name}/${ckpt_setting}-${expert_data_num} \
    --seed ${seed} \
    --temporal_agg true
