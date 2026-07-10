#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DIR="${ROOT_DIR}/external/robotwin_local"
ACT_DIR="${ROBOTWIN_DIR}/policy/ACT"
INTERACT_DIR="${ROBOTWIN_DIR}/policy/inter-act"
ENV_NAME="${ENV_NAME:-troncamp_env}"
TRONCAMP_SERVER="${TRONCAMP_SERVER:-https://submit.troncamp-mani.limxdynamics.com}"

info() {
  printf '[info] %s\n' "$*"
}

warn() {
  printf '[warn] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

require_path() {
  [ -e "$1" ] || die "缺少路径: $1"
}

require_robotwin() {
  [ -d "${ROBOTWIN_DIR}" ] || die "缺少官方包 ${ROBOTWIN_DIR}。请把 robotwin_local 放到 external/ 后重试。"
}

require_act() {
  require_robotwin
  [ -d "${ACT_DIR}" ] || die "缺少 ACT 目录: ${ACT_DIR}"
}

require_interact() {
  require_robotwin
  [ -d "${INTERACT_DIR}" ] || die "缺少 inter-act 目录: ${INTERACT_DIR}。请先执行: rsync -a policies/inter-act/ external/robotwin_local/policy/inter-act/"
}

activate_conda_env() {
  require_cmd conda
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

set_track_vars() {
  case "${1:-}" in
    T1|t1)
      TRACK="T1"
      TASK_NAME="adjust_bottle"
      TASK_CONFIG="adjust_bottle_200ep"
      CLEAN_CONFIG="adjust_bottle_clean"
      EXPERT_DATA_NUM="200"
      ;;
    T2|t2)
      TRACK="T2"
      TASK_NAME="grab_roller"
      TASK_CONFIG="grab_roller_400ep"
      CLEAN_CONFIG="grab_roller_clean"
      EXPERT_DATA_NUM="400"
      ;;
    T3|t3)
      TRACK="T3"
      TASK_NAME="stack_bowls_two"
      TASK_CONFIG="stack_bowls_two_200ep"
      CLEAN_CONFIG="stack_bowls_two_clean"
      EXPERT_DATA_NUM="200"
      ;;
    T4|t4)
      TRACK="T4"
      TASK_NAME="stack_bowls_three"
      TASK_CONFIG="stack_bowls_three_200ep"
      CLEAN_CONFIG="stack_bowls_three_clean"
      EXPERT_DATA_NUM="200"
      ;;
    *)
      die "未知赛道: ${1:-空}，可选 T1/T2/T3/T4"
      ;;
  esac
}

task_config_path() {
  printf '%s/task_config/%s.yml\n' "${ROBOTWIN_DIR}" "${TASK_CONFIG}"
}

default_ckpt_dir() {
  printf '%s/act_ckpt/act-%s/%s-%s\n' "${ACT_DIR}" "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}"
}

default_best_ckpt() {
  printf '%s/policy_best.ckpt\n' "$(default_ckpt_dir)"
}

default_interact_ckpt_dir() {
  printf '%s/inter_act_ckpt/inter-act-%s/%s-%s\n' "${INTERACT_DIR}" "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}"
}

default_interact_best_ckpt() {
  printf '%s/policy_best.ckpt\n' "$(default_interact_ckpt_dir)"
}
