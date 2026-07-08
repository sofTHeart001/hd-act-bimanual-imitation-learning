#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

missing=0

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    info "$1: $(command -v "$1")"
  else
    warn "缺少命令: $1"
    missing=1
  fi
}

check_path() {
  if [ -e "$1" ]; then
    info "存在: $1"
  else
    warn "缺少: $1"
    missing=1
  fi
}

check_cmd conda
check_cmd ffmpeg
check_cmd python3

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  info "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
else
  warn "nvidia-smi 当前不可用；采集/训练/评测需要可用 NVIDIA 驱动。"
fi

env_exists=0
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  info "conda 环境已存在: ${ENV_NAME}"
  env_exists=1
else
  warn "conda 环境未创建: ${ENV_NAME}，可执行 make env"
fi

check_path "${ROBOTWIN_DIR}"
check_path "${ROBOTWIN_DIR}/script/_install.sh"
check_path "${ACT_DIR}/process_data.sh"
check_path "${ACT_DIR}/train.sh"
check_path "${ACT_DIR}/eval.sh"
check_path "${ROBOTWIN_DIR}/task_config/adjust_bottle_200ep.yml"

if [ -f "${ROOT_DIR}/setup/env_check.py" ]; then
  info "官方 env_check.py 存在"
else
  warn "官方 setup/env_check.py 不存在，将使用 setup/local_env_check.py 做基础检查。"
fi

if [ "${env_exists}" -eq 1 ]; then
  conda run -n "${ENV_NAME}" python "${ROOT_DIR}/setup/local_env_check.py" || missing=1
else
  python3 "${ROOT_DIR}/setup/local_env_check.py" || missing=1
fi

exit "${missing}"
