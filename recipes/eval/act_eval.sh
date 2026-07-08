#!/bin/bash
# Phase 4 · Tron2 ACT evaluation
# Usage:
#   bash recipes/eval/act_eval.sh <task_name> <task_config> <expert_data_num> <test_num> <seed> <gpu_id>
#   bash recipes/eval/act_eval.sh pick_dual_bottles tron2_50ep 50 5 0 0
# Writes seed_manifest.jsonl next to RoboTwin _result.txt via run_act_eval.py.

set -euo pipefail

task_name=${1:-pick_dual_bottles}
task_config=${2:-tron2_50ep}
expert_data_num=${3:-50}
test_num=${4:-5}
seed=${5:-0}
gpu_id=${6:-0}
state_dim=${TRON2_ACT_STATE_DIM:-16}
ckpt_name=${TRON2_ACT_CKPT_NAME:-policy_best.ckpt}
ckpt_setting=${TRON2_ACT_CKPT_SETTING:-${task_config}-${expert_data_num}}
eval_video_log=${TRON2_ACT_EVAL_VIDEO_LOG:-False}
expert_check=${TRON2_ACT_EXPERT_CHECK:-True}
temporal_agg=${TRON2_ACT_TEMPORAL_AGG:-True}

SUBREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROBOTWIN_ROOT="${TRON2_ROBOTWIN_DIR:-${SUBREPO_ROOT}/external/robotwin_local}"
PYTHON_BIN="${PYTHON:-python3}"
ckpt_dir=${TRON2_ACT_CKPT_DIR:-policy/ACT/act_ckpt/tron2-${task_name}/${ckpt_setting}}

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
    echo "[!] 项目私有 RoboTwin runtime 不存在: ${ROBOTWIN_ROOT}"
    echo "    先运行: bash recipes/rollout/bootstrap_robotwin_local.sh"
    exit 1
fi

"${PYTHON_BIN}" "${SUBREPO_ROOT}/recipes/train/patch_act_runtime.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --state-dim "${state_dim}"

cd "${ROBOTWIN_ROOT}"

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${SUBREPO_ROOT}/recipes/eval/run_act_eval.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    -- \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --ckpt_dir "${ckpt_dir}" \
    --ckpt_name "${ckpt_name}" \
    --state_dim "${state_dim}" \
    --seed "${seed}" \
    --temporal_agg "${temporal_agg}" \
    --test_num "${test_num}" \
    --eval_video_log "${eval_video_log}" \
    --expert_check "${expert_check}"
