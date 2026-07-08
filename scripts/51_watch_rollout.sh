#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

set_track_vars "${1:-}"
SEED="${2:-0}"
CKPT_DIR="${3:-$(default_ckpt_dir)}"

require_robotwin
require_path "${ROOT_DIR}/starter/watch_rollout.py"
require_path "${CKPT_DIR}"

info "单场景核查 ${TRACK}: seed=${SEED} ckpt_dir=${CKPT_DIR}"
(cd "${ROOT_DIR}" && python starter/watch_rollout.py --track "${TRACK}" --ckpt-dir "${CKPT_DIR}" --seed "${SEED}")

