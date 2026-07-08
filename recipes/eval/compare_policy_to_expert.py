#!/usr/bin/env python3
"""Compare ACT closed-loop policy traces against processed expert actions.

This is a small numeric diagnostic. It reads an `act_seed_matrix.py` summary,
loads each referenced `trace.jsonl`, aligns the policy actions with the
processed expert episode by timestep, and writes arm/gripper L1 deltas.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARM_INDICES = [idx for idx in range(16) if idx not in (7, 15)]
GRIPPER_INDICES = [7, 15]


def subrepo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    repo = subrepo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", required=True, help="Path to an act_seed_matrix.py JSON summary.")
    parser.add_argument("--robotwin-root", default=str(repo / "external" / "robotwin_local"))
    parser.add_argument("--task-name", default="pick_dual_bottles")
    parser.add_argument("--task-config", default="tron2_100ep")
    parser.add_argument("--expert-data-num", type=int, default=100)
    parser.add_argument("--output-dir", default=str(repo / "results" / "eval" / "policy_expert_compare"))
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def load_policy_actions(trace_jsonl: Path) -> np.ndarray:
    rows = [json.loads(line) for line in trace_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty trace: {trace_jsonl}")
    return np.asarray([row["action"] for row in rows], dtype=np.float32)


def load_expert_actions(robotwin_root: Path, task_name: str, task_config: str, expert_data_num: int, episode_index: int) -> np.ndarray:
    episode_path = (
        robotwin_root
        / "policy"
        / "ACT"
        / "processed_data"
        / f"sim-{task_name}"
        / f"{task_config}-{expert_data_num}"
        / f"episode_{episode_index}.hdf5"
    )
    with h5py.File(episode_path, "r") as root:
        return np.asarray(root["/action"], dtype=np.float32)


def compare_row(row: dict[str, Any], robotwin_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = resolve(row["summary_path"])
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    trace_jsonl = resolve(summary["trace_jsonl"])

    policy = load_policy_actions(trace_jsonl)
    expert = load_expert_actions(
        robotwin_root,
        args.task_name,
        args.task_config,
        args.expert_data_num,
        int(row["episode_index"]),
    )
    n = min(len(policy), len(expert))
    if n <= 0:
        raise RuntimeError(f"cannot align empty policy/expert arrays for {summary_path}")

    arm_l1 = np.mean(np.abs(policy[:n, ARM_INDICES] - expert[:n, ARM_INDICES]), axis=1)
    gripper_l1 = np.mean(np.abs(policy[:n, GRIPPER_INDICES] - expert[:n, GRIPPER_INDICES]), axis=1)
    n80 = min(n, 80)

    return {
        "split": row.get("split"),
        "episode_index": int(row["episode_index"]),
        "seed": int(row["seed"]),
        "success": bool(row["success"]),
        "aligned_steps": int(n),
        "policy_steps": int(len(policy)),
        "expert_steps": int(len(expert)),
        "mean_arm_l1_first80": float(np.mean(arm_l1[:n80])),
        "mean_arm_l1_all": float(np.mean(arm_l1)),
        "mean_gripper_l1_all": float(np.mean(gripper_l1)),
        "final_arm_l1": float(arm_l1[n - 1]),
        "final_gripper_l1": float(gripper_l1[n - 1]),
        "final_policy_gripper": float(np.mean(policy[n - 1, GRIPPER_INDICES])),
        "final_expert_gripper": float(np.mean(expert[n - 1, GRIPPER_INDICES])),
        "summary_path": str(summary_path),
        "trace_jsonl": str(trace_jsonl),
    }


def main() -> int:
    args = parse_args()
    matrix_path = resolve(args.matrix_json)
    robotwin_root = resolve(args.robotwin_root)

    with matrix_path.open(encoding="utf-8") as f:
        matrix = json.load(f)

    rows = [compare_row(row, robotwin_root, args) for row in matrix["rows"]]
    result = {"matrix_json": str(matrix_path), "args": vars(args), "rows": rows}

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.task_name}_{args.task_config}_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"[compare-policy-to-expert] summary={out_path}")
    print("split episode seed success aligned mean_arm_l1_all final_arm_l1 final_policy_gripper final_expert_gripper")
    for row in rows:
        print(
            row["split"],
            row["episode_index"],
            row["seed"],
            row["success"],
            row["aligned_steps"],
            f"{row['mean_arm_l1_all']:.4f}",
            f"{row['final_arm_l1']:.4f}",
            f"{row['final_policy_gripper']:.4f}",
            f"{row['final_expert_gripper']:.4f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
