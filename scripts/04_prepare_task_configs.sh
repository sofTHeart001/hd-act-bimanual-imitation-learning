#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_robotwin

src="${ROBOTWIN_DIR}/task_config/adjust_bottle_200ep.yml"
require_path "${src}"

for cfg in grab_roller_400ep stack_bowls_two_200ep stack_bowls_three_200ep; do
  dst="${ROBOTWIN_DIR}/task_config/${cfg}.yml"
  if [ -f "${dst}" ]; then
    info "配置已存在，跳过: ${dst}"
  else
    cp "${src}" "${dst}"
    info "已从 adjust_bottle_200ep.yml 复制模板: ${dst}"
  fi
done

warn "T2-T4 配置已生成为模板，请按任务调 episode_num、场景和域随机化参数。"
