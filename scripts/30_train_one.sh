#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
SEED="${2:-0}"
GPU_ID="${3:-0}"

require_act
require_path "${ACT_DIR}/train.sh"

info "训练 ${TRACK}: ${TASK_NAME} ${TASK_CONFIG} ${EXPERT_DATA_NUM} seed=${SEED} GPU=${GPU_ID}"
(cd "${ACT_DIR}" && bash train.sh "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}" "${SEED}" "${GPU_ID}")

