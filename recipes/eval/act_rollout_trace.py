#!/usr/bin/env python3
"""Trace one Tron2 ACT closed-loop rollout inside project-local RoboTwin.

The trace is intentionally numeric and small: it records action deltas,
drive-target movement, real-qpos movement, drive tracking error, TCP movement,
and success state. Large videos stay on the server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--ckpt-setting", default=os.environ.get("TRON2_ACT_CKPT_SETTING"))
    parser.add_argument("--ckpt-name", default=os.environ.get("TRON2_ACT_CKPT_NAME", "policy_best.ckpt"))
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=-1, help="Matches RoboTwin eval_policy.py seed convention; -1 starts at seed 0.")
    parser.add_argument("--absolute-seed", type=int, default=None, help="Use an exact RoboTwin seed instead of eval_policy.py seed convention.")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--temporal-agg", choices=("True", "False", "true", "false"), default=os.environ.get("TRON2_ACT_TEMPORAL_AGG", "True"))
    parser.add_argument(
        "--gripper-intervention",
        choices=("none", "clamp-zero-after-below", "clamp-negative-after-below"),
        default="none",
        help="Trace-only diagnostic override. Never use for accepted policy metrics.",
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-patience", type=int, default=1)
    parser.add_argument("--output-dir", default=str(subrepo_root() / "results" / "eval" / "trace"))
    return parser.parse_args()


def resolve_local_runtime(robotwin_root: Path) -> Path:
    root = robotwin_root.expanduser().resolve(strict=False)
    shared = (subrepo_root() / "external" / "robotwin").resolve(strict=False)
    if root == shared or shared in root.parents:
        raise SystemExit(f"refusing shared upstream RoboTwin checkout: {root}")
    if not (root / "script" / "eval_policy.py").is_file():
        raise SystemExit(f"RoboTwin eval_policy.py not found under: {root}")
    return root


def as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def mean_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def tcp_pose(task_env: Any, arm: str) -> np.ndarray:
    getter = task_env.robot.get_left_tcp_pose if arm == "left" else task_env.robot.get_right_tcp_pose
    return as_float_array(getter())


def real_qpos(task_env: Any) -> np.ndarray:
    left = task_env.robot.get_left_arm_real_jointState()
    right = task_env.robot.get_right_arm_real_jointState()
    return as_float_array(left + right)


def drive_qpos_from_observation(observation: dict[str, Any]) -> np.ndarray:
    return as_float_array(observation["joint_action"]["vector"])


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def bool_arg(value: str) -> bool:
    return value.lower() == "true"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "cmd_delta_l1",
        "cmd_delta_max",
        "drive_move_l1",
        "real_move_l1",
        "drive_tracking_l1",
        "left_tcp_move",
        "right_tcp_move",
    ]
    summary: dict[str, Any] = {"steps": len(rows)}
    for key in numeric_keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        summary[key] = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "p90": float(np.quantile(arr, 0.9)) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
        }
    if rows:
        summary["final_step"] = rows[-1]["step"]
        summary["final_eval_success"] = bool(rows[-1]["eval_success"])
        summary["final_left_gripper_cmd"] = rows[-1]["left_gripper_cmd"]
        summary["final_right_gripper_cmd"] = rows[-1]["right_gripper_cmd"]
    return summary


def patch_act_runtime(robotwin_root: Path, state_dim: int) -> None:
    from recipes.train.patch_act_runtime import (
        ensure_local_runtime,
        patch_act_policy,
        patch_deploy_config,
        patch_detr_vae,
        patch_eval_policy,
        patch_imitate,
        patch_utils_alignment,
    )

    act_dir = ensure_local_runtime(robotwin_root)
    patch_imitate(act_dir)
    patch_utils_alignment(act_dir)
    patch_detr_vae(act_dir)
    patch_act_policy(act_dir)
    patch_eval_policy(act_dir)
    patch_deploy_config(act_dir, state_dim)


def maybe_intervene_gripper(
    action: np.ndarray,
    step: int,
    below_count: int,
    intervention_start_step: int | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int | None, bool]:
    if args.gripper_intervention == "none":
        return action, 0, None, False

    avg_gripper_cmd = float((action[7] + action[15]) / 2.0)
    if intervention_start_step is None:
        below_count = below_count + 1 if avg_gripper_cmd < args.gripper_threshold else 0
        if below_count >= args.gripper_patience:
            intervention_start_step = step

    intervention_active = intervention_start_step is not None
    if intervention_active:
        clamp_value = 0.0 if args.gripper_intervention == "clamp-zero-after-below" else -0.02
        action = action.copy()
        action[7] = clamp_value
        action[15] = clamp_value
    return action, below_count, intervention_start_step, intervention_active


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    repo = subrepo_root()
    robotwin_root = resolve_local_runtime(Path(args.robotwin_root))
    act_dir = robotwin_root / "policy" / "ACT"
    ckpt_setting = args.ckpt_setting or f"{args.task_config}-{args.expert_data_num}"
    ckpt_dir = Path("policy") / "ACT" / "act_ckpt" / f"tron2-{args.task_name}" / ckpt_setting

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "script"))
    sys.path.insert(0, str(act_dir))
    os.chdir(robotwin_root)

    patch_act_runtime(robotwin_root, args.state_dim)

    from recipes.rollout.tron2_runtime_patch import apply_tron2_runtime_patches
    from recipes.rollout.tron2_obs_force_patch import apply_obs_force_patch

    def _resolve_pot_links(task):
        pot = getattr(task, "pot", None)
        if pot is None:
            return None
        actor = getattr(pot, "actor", None)
        if actor is None:
            return None
        try:
            return [link.entity for link in actor.get_links()]
        except Exception:
            return None

    apply_tron2_runtime_patches()
    apply_obs_force_patch(get_pot_entity=_resolve_pot_links)

    try:
        from test_render import Sapien_TEST

        Sapien_TEST()
    except Exception as exc:
        print(f"[act-rollout-trace] render preflight skipped: {exc}", flush=True)

    from envs import CONFIGS_PATH
    from script.eval_policy import class_decorator, get_embodiment_config
    from policy.ACT.deploy_policy import encode_obs, get_model, reset_model

    with Path(f"task_config/{args.task_config}.yml").open(encoding="utf-8") as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args["task_name"] = args.task_name
    task_args["task_config"] = args.task_config
    task_args["ckpt_setting"] = ckpt_setting

    with Path(CONFIGS_PATH + "_embodiment_config.yml").open(encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = task_args["embodiment"]
    if len(embodiment_type) != 1:
        raise SystemExit("act_rollout_trace.py currently expects one dual-arm embodiment entry.")

    robot_file = embodiment_types[embodiment_type[0]]["file_path"]
    task_args["left_robot_file"] = robot_file
    task_args["right_robot_file"] = robot_file
    task_args["dual_arm_embodied"] = True
    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["policy_name"] = "ACT"
    task_args["eval_mode"] = True
    task_args["eval_video_log"] = False
    task_args["eval_video_save_dir"] = None

    with (act_dir / "deploy_policy.yml").open(encoding="utf-8") as f:
        usr_args = yaml.safe_load(f)

    usr_args.update(
        {
            "task_name": args.task_name,
            "policy_name": "ACT",
            "task_config": args.task_config,
            "ckpt_setting": ckpt_setting,
            "seed": args.seed,
            "instruction_type": "unseen",
            "ckpt_dir": str(ckpt_dir),
            "ckpt_name": args.ckpt_name,
            "state_dim": args.state_dim,
            "action_dim": args.state_dim,
            "temporal_agg": bool_arg(args.temporal_agg),
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
            "left_arm_dim": len(task_args["left_embodiment_config"]["arm_joints_name"][0]),
            "right_arm_dim": len(task_args["right_embodiment_config"]["arm_joints_name"][1]),
            "device": "cuda:0",
        }
    )

    task_env = class_decorator(args.task_name)
    seed = args.absolute_seed if args.absolute_seed is not None else 100000 * (1 + args.seed)
    model = get_model(usr_args)

    out_root = Path(args.output_dir).expanduser().resolve(strict=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{args.task_name}_{args.task_config}_seed{seed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trace.jsonl"
    summary_path = out_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)
        task_env.set_instruction(instruction="")
        reset_model(model)

        gripper_below_count = 0
        gripper_intervention_start_step: int | None = None

        with jsonl_path.open("w", encoding="utf-8") as f:
            for step in range(min(args.max_steps, task_env.step_lim)):
                observation = task_env.get_obs()
                encoded = encode_obs(observation)
                before_drive = drive_qpos_from_observation(observation)
                before_real = real_qpos(task_env)
                before_left_tcp = tcp_pose(task_env, "left")
                before_right_tcp = tcp_pose(task_env, "right")

                actions = np.atleast_2d(np.asarray(model.get_action(encoded), dtype=np.float32))
                raw_action = actions[0].copy()
                action, gripper_below_count, gripper_intervention_start_step, gripper_intervention_active = (
                    maybe_intervene_gripper(raw_action, step, gripper_below_count, gripper_intervention_start_step, args)
                )
                task_env.take_action(action)

                after_observation = task_env.get_obs()
                after_drive = drive_qpos_from_observation(after_observation)
                after_real = real_qpos(task_env)
                after_left_tcp = tcp_pose(task_env, "left")
                after_right_tcp = tcp_pose(task_env, "right")

                row = {
                    "step": step,
                    "seed": seed,
                    "eval_success": bool(task_env.eval_success),
                    "take_action_cnt": int(task_env.take_action_cnt),
                    "cmd_delta_l1": mean_l1(action, before_drive),
                    "cmd_delta_max": max_abs(action, before_drive),
                    "drive_move_l1": mean_l1(after_drive, before_drive),
                    "real_move_l1": mean_l1(after_real, before_real),
                    "drive_tracking_l1": mean_l1(after_real, after_drive),
                    "left_tcp_move": float(np.linalg.norm(after_left_tcp[:3] - before_left_tcp[:3])),
                    "right_tcp_move": float(np.linalg.norm(after_right_tcp[:3] - before_right_tcp[:3])),
                    "left_gripper_before": float(before_drive[7]),
                    "raw_left_gripper_cmd": float(raw_action[7]),
                    "left_gripper_cmd": float(action[7]),
                    "left_gripper_after": float(after_drive[7]),
                    "right_gripper_before": float(before_drive[15]),
                    "raw_right_gripper_cmd": float(raw_action[15]),
                    "right_gripper_cmd": float(action[15]),
                    "right_gripper_after": float(after_drive[15]),
                    "gripper_intervention_active": bool(gripper_intervention_active),
                    "gripper_intervention_start_step": gripper_intervention_start_step,
                    "action": action,
                    "raw_action": raw_action,
                    "before_drive_qpos": before_drive,
                    "after_drive_qpos": after_drive,
                    "before_real_qpos": before_real,
                    "after_real_qpos": after_real,
                    "before_left_tcp": before_left_tcp,
                    "after_left_tcp": after_left_tcp,
                    "before_right_tcp": before_right_tcp,
                    "after_right_tcp": after_right_tcp,
                }
                rows.append(row)
                f.write(json.dumps({key: json_ready(value) for key, value in row.items()}, ensure_ascii=True) + "\n")
                f.flush()
                if task_env.eval_success:
                    break
    finally:
        try:
            task_env.close_env()
        except Exception:
            pass

    summary = summarize(rows)
    summary.update(
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "ckpt_setting": ckpt_setting,
            "ckpt_name": args.ckpt_name,
            "temporal_agg": bool_arg(args.temporal_agg),
            "gripper_intervention": args.gripper_intervention,
            "gripper_threshold": args.gripper_threshold,
            "gripper_patience": args.gripper_patience,
            "gripper_intervention_start_step": next(
                (row["gripper_intervention_start_step"] for row in rows if row.get("gripper_intervention_active")),
                None,
            ),
            "seed": seed,
            "trace_jsonl": str(jsonl_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"[act-rollout-trace] trace={jsonl_path}")
    print(f"[act-rollout-trace] summary={summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
