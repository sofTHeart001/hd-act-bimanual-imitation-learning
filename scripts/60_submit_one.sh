#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
CKPT="${2:-$(default_best_ckpt)}"

require_robotwin
require_path "${ROOT_DIR}/submit/submit.py"
require_path "${CKPT}"

cmd=(python submit/submit.py --server "${TRONCAMP_SERVER}" --track "${TRACK}" --ckpt "${CKPT}")

if [ "${TRACK}" != "T1" ]; then
  cmd+=(--code-dir "${ROBOTWIN_DIR}")
fi

if [ -n "${TRONCAMP_TOKEN_FILE:-}" ]; then
  cmd+=(--token-file "${TRONCAMP_TOKEN_FILE}")
elif [ -n "${TRONCAMP_TOKEN:-}" ]; then
  :
elif [ -n "${TOKEN:-}" ]; then
  cmd+=(--token="${TOKEN}")
else
  warn "未设置 TRONCAMP_TOKEN/TRONCAMP_TOKEN_FILE/TOKEN，submit.py 可能会因缺 token 失败。"
fi

info "提交 ${TRACK}: ckpt=${CKPT}"
(cd "${ROOT_DIR}" && "${cmd[@]}")

