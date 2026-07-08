#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"

require_act
require_path "${ACT_DIR}/process_data.sh"

info "处理数据 ${TRACK}: ${TASK_NAME} ${TASK_CONFIG} ${EXPERT_DATA_NUM}"
(cd "${ACT_DIR}" && bash process_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}")

