#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
CKPT_DIR="${2:-$(default_ckpt_dir)}"

require_robotwin
require_path "${ROOT_DIR}/starter/eval_local.py"
require_path "${CKPT_DIR}"

info "本地自评 ${TRACK}: ckpt_dir=${CKPT_DIR}"
(cd "${ROOT_DIR}" && python starter/eval_local.py --track "${TRACK}" --ckpt-dir "${CKPT_DIR}")

