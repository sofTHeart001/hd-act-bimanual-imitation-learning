"""Collect RoboTwin data for Tron2 (ACT 16-D).

只打 Tron2 自碰撞 runtime patch(保证规划-跟踪专家不自撞),用标准 RoboTwin
采集产出 16-D obs/action 的 demo(无力信号——ACT 四任务套餐不使用力)。
专家 = envs/<task_name>.py 的 play_once();改进专家直接改它。
"""

from __future__ import annotations

import argparse
import os
import sys


SUBREPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SUBREPO)

from recipes.rollout.robotwin_runtime import get_robotwin_dir

ROBOTWIN = get_robotwin_dir(SUBREPO)
sys.path.insert(0, ROBOTWIN)
os.chdir(ROBOTWIN)

from recipes.rollout.tron2_runtime_patch import apply_tron2_runtime_patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    args = parser.parse_args()

    from script.test_render import Sapien_TEST

    Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    apply_tron2_runtime_patches()

    from script.collect_data import main as collect_main

    collect_main(task_name=args.task_name, task_config=args.task_config)


if __name__ == "__main__":
    main()
