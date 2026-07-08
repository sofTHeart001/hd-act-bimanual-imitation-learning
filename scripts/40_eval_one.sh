#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
MODE="${2:-clean}"
SEED="${3:-0}"
GPU_ID="${4:-0}"
CKPT_SETTING="${5:-${TASK_CONFIG}}"

require_act
require_path "${ACT_DIR}/eval.sh"

case "${MODE}" in
  same)
    EVAL_CONFIG="${TASK_CONFIG}"
    ;;
  clean)
    EVAL_CONFIG="${CLEAN_CONFIG}"
    ;;
  *)
    EVAL_CONFIG="${MODE}"
    ;;
esac

info "评测 ${TRACK}: env=${EVAL_CONFIG} ckpt_setting=${CKPT_SETTING} num=${EXPERT_DATA_NUM} seed=${SEED} GPU=${GPU_ID}"
(cd "${ACT_DIR}" && bash eval.sh "${TASK_NAME}" "${EVAL_CONFIG}" "${CKPT_SETTING}" "${EXPERT_DATA_NUM}" "${SEED}" "${GPU_ID}")

