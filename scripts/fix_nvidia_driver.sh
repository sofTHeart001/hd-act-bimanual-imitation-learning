#!/usr/bin/env bash

set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "[error] 请用 sudo 执行: sudo bash scripts/fix_nvidia_driver.sh" >&2
  exit 1
fi

kernel="$(uname -r)"
module_pkg="linux-modules-nvidia-580-server-open-${kernel}"
meta_pkg="linux-modules-nvidia-580-server-open-generic-hwe-22.04"
driver_pkg="nvidia-driver-580-server-open"

echo "[info] current kernel: ${kernel}"
echo "[info] installing: ${module_pkg} ${meta_pkg} ${driver_pkg}"

apt-get update
apt-get install -y "${module_pkg}" "${meta_pkg}" "${driver_pkg}"

echo "[info] refreshing module dependency cache"
depmod -a "${kernel}"

echo "[info] trying to load NVIDIA modules"
modprobe nvidia || true
modprobe nvidia_uvm || true
modprobe nvidia_drm || true

echo "[info] NVIDIA module status:"
lsmod | grep -E '^nvidia' || true

echo "[info] nvidia-smi:"
if nvidia-smi; then
  echo "[ok] NVIDIA driver is active."
else
  echo "[warn] nvidia-smi still failed. Reboot, then run: nvidia-smi" >&2
  exit 2
fi

