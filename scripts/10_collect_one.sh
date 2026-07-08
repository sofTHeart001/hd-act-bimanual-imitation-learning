#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
GPU_ID="${2:-0}"

require_robotwin
require_path "${ROOT_DIR}/collect_data.sh"
require_path "$(task_config_path)"

info "采集 ${TRACK}: ${TASK_NAME} ${TASK_CONFIG} GPU=${GPU_ID}"
(cd "${ROOT_DIR}" && bash collect_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${GPU_ID}")

