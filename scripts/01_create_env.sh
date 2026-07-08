#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_cmd conda

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  info "conda 环境已存在: ${ENV_NAME}"
else
  info "创建 conda 环境: ${ENV_NAME} (python=3.10)"
  conda create -y -n "${ENV_NAME}" python=3.10
fi

activate_conda_env
python -m pip install --upgrade pip
python -m pip install "setuptools<81"

info "Python: $(python --version)"
info "pip: $(python -m pip --version)"
python -m pip show setuptools | sed -n '1,3p'

