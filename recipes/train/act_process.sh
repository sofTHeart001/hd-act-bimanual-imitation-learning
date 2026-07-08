#!/bin/bash
# Process RoboTwin expert HDF5 episodes into ACT training HDF5 files.
#
# Usage:
#   bash recipes/train/act_process.sh <task_name> <task_config> <expert_data_num>
#   bash recipes/train/act_process.sh pick_dual_bottles tron2_100ep 100

set -euo pipefail

task_name=${1:-pick_dual_bottles}
task_config=${2:-tron2_50ep}
expert_data_num=${3:-50}
state_dim=${TRON2_ACT_STATE_DIM:-16}

SUBREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROBOTWIN_ROOT="${TRON2_ROBOTWIN_DIR:-${SUBREPO_ROOT}/external/robotwin_local}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
    echo "[act-process] ERROR: project-local RoboTwin runtime not found: ${ROBOTWIN_ROOT}" >&2
    echo "              Run: bash recipes/rollout/bootstrap_robotwin_local.sh" >&2
    exit 2
fi

ACT_DIR="${ROBOTWIN_ROOT}/policy/ACT"
if [[ ! -d "${ACT_DIR}" ]]; then
    echo "[act-process] ERROR: ACT runtime directory not found: ${ACT_DIR}" >&2
    exit 2
fi

robotwin_real="$(cd "${ROBOTWIN_ROOT}" && pwd -P)"
shared_real=""
if [[ -e "${SUBREPO_ROOT}/external/robotwin" ]]; then
    shared_real="$(cd "${SUBREPO_ROOT}/external/robotwin" && pwd -P)"
fi

if [[ -n "${shared_real}" ]]; then
    case "${robotwin_real}" in
        "${shared_real}"|"${shared_real}"/*)
            echo "[act-process] ERROR: refusing shared upstream RoboTwin checkout: ${robotwin_real}" >&2
            exit 2
            ;;
    esac
fi

raw_data_dir="${ROBOTWIN_ROOT}/data/${task_name}/${task_config}/data"
if [[ ! -d "${raw_data_dir}" ]]; then
    echo "[act-process] ERROR: raw expert data not found: ${raw_data_dir}" >&2
    exit 2
fi

raw_count="$(find "${raw_data_dir}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l | tr -d ' ')"
if (( raw_count < expert_data_num )); then
    echo "[act-process] ERROR: raw expert data has ${raw_count} episodes, expected ${expert_data_num}" >&2
    exit 2
fi

cd "${ACT_DIR}"
"${PYTHON_BIN}" process_data.py "${task_name}" "${task_config}" "${expert_data_num}"

"${PYTHON_BIN}" "${SUBREPO_ROOT}/recipes/train/validate_act_shapes.py" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --task-name "${task_name}" \
    --task-config "${task_config}" \
    --expert-data-num "${expert_data_num}" \
    --state-dim "${state_dim}"
