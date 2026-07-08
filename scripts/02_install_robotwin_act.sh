#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_robotwin
activate_conda_env

info "锁定 setuptools<81"
python -m pip install "setuptools<81"

require_path "${ROBOTWIN_DIR}/script/_install.sh"
info "安装 RoboTwin 本体"
(cd "${ROBOTWIN_DIR}" && bash script/_install.sh)

if [ -f "${ROOT_DIR}/setup/requirements.txt" ]; then
  info "安装 ACT 训练栈和 cuRobo runtime 依赖"
  python -m pip install -r "${ROOT_DIR}/setup/requirements.txt"
else
  warn "未找到 ${ROOT_DIR}/setup/requirements.txt，跳过官方依赖安装。请确认选手包是否完整。"
fi

require_path "${ROBOTWIN_DIR}/envs/curobo"
info "安装 cuRobo 0.8.0"
SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 python -m pip install -e "${ROBOTWIN_DIR}/envs/curobo" --no-build-isolation

bash "${ROOT_DIR}/scripts/03_restore_paths.sh"

if ffmpeg -version >/dev/null 2>&1; then
  info "ffmpeg 已存在"
else
  info "安装 imageio-ffmpeg 兜底"
  python -m pip install imageio-ffmpeg
  ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "${CONDA_PREFIX}/bin/ffmpeg"
fi

if [ -f "${ROOT_DIR}/setup/env_check.py" ]; then
  info "运行官方环境自检"
  python "${ROOT_DIR}/setup/env_check.py"
else
  warn "官方 env_check.py 不存在，运行本地基础自检"
  python "${ROOT_DIR}/setup/local_env_check.py"
fi

