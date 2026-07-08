#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

count=0
while IFS= read -r file; do
  sed -i "s#__KIT_ROOT__#${ROOT_DIR}#g" "$file"
  count=$((count + 1))
done < <(grep -rl --include='*.yml' '__KIT_ROOT__' "${ROOT_DIR}" || true)

info "已还原 __KIT_ROOT__ 的 yml 文件数量: ${count}"

