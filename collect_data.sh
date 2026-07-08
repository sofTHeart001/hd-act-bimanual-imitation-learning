#!/usr/bin/env bash
# collect_data.sh <task_name> <task_config> [gpu_id] —— 用可改专家自采 RoboTwin demo(T2-T4 自采路径)
# 命名/签名对齐 RoboTwin 官方 collect_data.sh(task_name + task_config + gpu_id)。
#
#   专家 = envs/<task_name>.py 的 play_once();想改进专家直接改它。
#   本脚本打 Tron2 自碰撞 runtime patch 后跑 RoboTwin 采集,只留成功 episode
#   (裸官方 collect_data.py 会臂自碰撞 → 专家 0%,故必须经此 wrapper)。
#   采集集数 / 域随机化 / 场景在 task_config(task_config/<task_config>.yml)里调。
#   采完用 ACT 训练栈的 process_data 转 16-D 训练格式:
#     cd external/robotwin_local/policy/ACT && bash process_data.sh <task_name> <task_config> <num>
#
# 例:bash collect_data.sh adjust_bottle adjust_bottle_200ep 0
set -u
if [ $# -lt 2 ]; then
  echo "用法: bash collect_data.sh <task_name> <task_config> [gpu_id]"
  echo "  如: bash collect_data.sh adjust_bottle adjust_bottle_200ep 0"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${3:-0}"
# 默认单趟采集(搜索期成功即存 HDF5,无回放阶段)。两阶段(搜索录关节路径 → 回放重放)会因
# curobo planner 构造重置 numpy 全局 RNG,让搜索期首集场景与回放期同 seed 场景不一致,专家换臂
# → 回放索引空关节路径崩溃。单趟天然免疫该 bug,也无空洞集问题。设 ROBOTWIN_SAVE_DURING_SEARCH=0
# 可回到两阶段采集。
export ROBOTWIN_SAVE_DURING_SEARCH="${ROBOTWIN_SAVE_DURING_SEARCH:-1}"
cd "$(dirname "$0")"
python recipes/rollout/collect_tron2_data.py "$1" "$2"
