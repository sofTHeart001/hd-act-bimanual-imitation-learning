#!/bin/bash
# Phase 3 · Tron2 ACT 训练
# 对比 RoboTwin 第一方 policy/ACT/train.sh：state_dim 14 → 16（tron2 双臂 7+1 DOF × 2）
# 用法：
#   bash recipes/train/act_train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id>
#   bash recipes/train/act_train.sh pick_dual_bottles tron2_50ep 50 0 0
#
# 前置条件：
#   1. recipes/rollout/tron2_smoke.yml 冒烟已过（curobo + embodiment 加载 OK）
#   2. 已用 recipes/rollout/collect_tron2_data.py 采集到 data/<task>/<config>/data/episode*.hdf5
#   3. 已跑 process_data.sh 转成 ACT pickle 格式

set -euo pipefail

task_name=${1:-pick_dual_bottles}
task_config=${2:-tron2_50ep}
expert_data_num=${3:-50}
seed=${4:-0}
gpu_id=${5:-0}
state_dim=${TRON2_ACT_STATE_DIM:-16}
ckpt_setting=${TRON2_ACT_CKPT_SETTING:-${task_config}-${expert_data_num}}
num_epochs=${TRON2_ACT_NUM_EPOCHS:-6000}
batch_size=${TRON2_ACT_BATCH_SIZE:-8}

SUBREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROBOTWIN_ROOT="${TRON2_ROBOTWIN_DIR:-${SUBREPO_ROOT}/external/robotwin_local}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
    echo "[!] 项目私有 RoboTwin runtime 不存在: ${ROBOTWIN_ROOT}"
    echo "    先运行: bash recipes/rollout/bootstrap_robotwin_local.sh"
    exit 1
fi

"${PYTHON_BIN}" "${SUBREPO_ROOT}/recipes/train/patch_act_runtime.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --state-dim "${state_dim}"

"${PYTHON_BIN}" "${SUBREPO_ROOT}/recipes/train/validate_act_shapes.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --task-name "${task_name}" \
    --task-config "${task_config}" \
    --expert-data-num "${expert_data_num}" \
    --state-dim "${state_dim}"

cd "${ROBOTWIN_ROOT}/policy/ACT"

export CUDA_VISIBLE_DEVICES=${gpu_id}

"${PYTHON_BIN}" imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/tron2-${task_name}/${ckpt_setting} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size "${batch_size}" \
    --dim_feedforward 3200 \
    --num_epochs "${num_epochs}" \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim "${state_dim}" \
    --seed ${seed}
