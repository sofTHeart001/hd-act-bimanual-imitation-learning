#!/usr/bin/env python3
"""Replay processed ACT actions through RoboTwin take_action for isolation.

This checks whether the processed HDF5 qpos/action stream is itself sufficient
to solve a task when fed back into the same eval control interface used by ACT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def subrepo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", default=os.environ.get("TRON2_ROBOTWIN_DIR", str(subrepo_root() / "external" / "robotwin_local")))
    parser.add_argument("--task-name", default="pick_dual_bottles")
    parser.add_argument("--task-config", default="tron2_50ep")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means replay the full processed action sequence.")
    parser.add_argument("--output-dir", default=str(subrepo_root() / "results" / "eval" / "replay"))
    return parser.parse_args()


def resolve_local_runtime(robotwin_root: Path) -> Path:
    root = robotwin_root.expanduser().resolve(strict=False)
    shared = (subrepo_root() / "external" / "robotwin").resolve(strict=False)
    if root == shared or shared in root.parents:
        raise SystemExit(f"refusing shared upstream RoboTwin checkout: {root}")
    if not (root / "script" / "eval_policy.py").is_file():
        raise SystemExit(f"RoboTwin eval_policy.py not found under: {root}")
    return root


def seed_for_episode(robotwin_root: Path, task_name: str, task_config: str, episode_index: int) -> int:
    seed_path = robotwin_root / "data" / task_name / task_config / "seed.txt"
    seeds = [int(token) for token in seed_path.read_text().split()]
    return seeds[episode_index]


def processed_episode_path(robotwin_root: Path, task_name: str, task_config: str, expert_data_num: int, episode_index: int) -> Path:
    return (
        robotwin_root
        / "policy"
        / "ACT"
        / "processed_data"
        / f"sim-{task_name}"
        / f"{task_config}-{expert_data_num}"
        / f"episode_{episode_index}.hdf5"
    )


def mean_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def drive_qpos(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)


def main() -> int:
    args = parse_args()
    repo = subrepo_root()
    robotwin_root = resolve_local_runtime(Path(args.robotwin_root))

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "script"))
    os.chdir(robotwin_root)

    from recipes.rollout.tron2_runtime_patch import apply_tron2_runtime_patches
    from envs import CONFIGS_PATH
    from script.eval_policy import class_decorator, get_embodiment_config

    apply_tron2_runtime_patches()

    with Path(f"task_config/{args.task_config}.yml").open(encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args["task_name"] = args.task_name
    task_args["task_config"] = args.task_config
    task_args["policy_name"] = "ACT"
    task_args["eval_mode"] = True
    task_args["eval_video_log"] = False
    task_args["eval_video_save_dir"] = None

    with Path(CONFIGS_PATH + "_embodiment_config.yml").open(encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = task_args["embodiment"]
    if len(embodiment_type) != 1:
        raise SystemExit("replay_processed_actions.py currently expects one dual-arm embodiment entry.")

    robot_file = embodiment_types[embodiment_type[0]]["file_path"]
    task_args["left_robot_file"] = robot_file
    task_args["right_robot_file"] = robot_file
    task_args["dual_arm_embodied"] = True
    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])

    seed = seed_for_episode(robotwin_root, args.task_name, args.task_config, args.episode_index)
    episode_path = processed_episode_path(robotwin_root, args.task_name, args.task_config, args.expert_data_num, args.episode_index)

    with h5py.File(episode_path, "r") as root:
        actions = np.asarray(root["/action"], dtype=np.float32)
        qpos = np.asarray(root["/observations/qpos"], dtype=np.float32)

    if args.max_steps > 0:
        actions = actions[: args.max_steps]
        qpos = qpos[: args.max_steps]

    task_env = class_decorator(args.task_name)
    rows: list[dict[str, Any]] = []
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
        task_env.set_instruction(instruction="")
        for step, action in enumerate(actions):
            before = drive_qpos(task_env.get_obs())
            reference = qpos[min(step, len(qpos) - 1)]
            task_env.take_action(action)
            after = drive_qpos(task_env.get_obs())
            rows.append(
                {
                    "step": step,
                    "cmd_to_before_l1": mean_l1(action, before),
                    "after_to_cmd_l1": mean_l1(after, action),
                    "before_to_reference_l1": mean_l1(before, reference),
                    "eval_success": bool(task_env.eval_success),
                }
            )
            if task_env.eval_success:
                break
    finally:
        try:
            task_env.close_env()
        except Exception:
            pass

    out_root = Path(args.output_dir).expanduser().resolve(strict=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{args.task_name}_{args.task_config}_episode{args.episode_index}_seed{seed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    rows_path = out_dir / "steps.jsonl"

    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "episode_index": args.episode_index,
        "seed": seed,
        "episode_path": str(episode_path),
        "steps_replayed": len(rows),
        "eval_success": bool(rows[-1]["eval_success"]) if rows else False,
        "cmd_to_before_l1_mean": float(np.mean([row["cmd_to_before_l1"] for row in rows])) if rows else 0.0,
        "before_to_reference_l1_mean": float(np.mean([row["before_to_reference_l1"] for row in rows])) if rows else 0.0,
        "rows_jsonl": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"[replay-processed-actions] summary={summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
