#!/usr/bin/env python3
"""Run a small train-vs-validation ACT rollout trace matrix.

This is a diagnostic driver around act_rollout_trace.py. It uses the same
numpy seed-1 train/validation split convention used by RoboTwin ACT loading,
then traces a few exact dataset train seeds and validation seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def subrepo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    repo = subrepo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="pick_dual_bottles")
    parser.add_argument("--task-config", default="tron2_50ep")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--ckpt-setting", default=os.environ.get("TRON2_ACT_CKPT_SETTING"))
    parser.add_argument("--ckpt-name", default=os.environ.get("TRON2_ACT_CKPT_NAME", "policy_best.ckpt"))
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--temporal-agg", choices=("True", "False", "true", "false"), default=os.environ.get("TRON2_ACT_TEMPORAL_AGG", "True"))
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--output-dir", default=str(repo / "results" / "eval" / "seed_matrix"))
    return parser.parse_args()


def load_dataset_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed_file is None:
        seed_path = (
            subrepo_root()
            / "external"
            / "robotwin_local"
            / "data"
            / args.task_name
            / args.task_config
            / "seed.txt"
        )
    else:
        seed_path = Path(args.seed_file)
    text = seed_path.expanduser().read_text(encoding="utf-8")
    seeds = [int(token) for token in text.split()]
    if len(seeds) < args.expert_data_num:
        raise SystemExit(f"seed file has {len(seeds)} seeds, expected at least {args.expert_data_num}: {seed_path}")
    return seeds[: args.expert_data_num]


def split_indices(num_items: int, train_ratio: float, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(split_seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    train_count = int(num_items * train_ratio)
    return indices[:train_count], indices[train_count:]


def latest_summary(trace_root: Path, task_name: str, task_config: str, seed: int) -> Path:
    matches = sorted(trace_root.glob(f"{task_name}_{task_config}_seed{seed}_*/summary.json"))
    if not matches:
        raise RuntimeError(f"no summary found for seed {seed} under {trace_root}")
    return matches[-1]


def run_trace(args: argparse.Namespace, seed: int) -> dict:
    trace_root = subrepo_root() / "results" / "eval" / "trace"
    cmd = [
        sys.executable,
        str(subrepo_root() / "recipes" / "eval" / "act_rollout_trace.py"),
        "--task-name",
        args.task_name,
        "--task-config",
        args.task_config,
        "--expert-data-num",
        str(args.expert_data_num),
        "--ckpt-setting",
        args.ckpt_setting,
        "--ckpt-name",
        args.ckpt_name,
        "--state-dim",
        str(args.state_dim),
        "--gpu-id",
        str(args.gpu_id),
        "--max-steps",
        str(args.max_steps),
        "--temporal-agg",
        args.temporal_agg,
        "--absolute-seed",
        str(seed),
    ]
    print("[act-seed-matrix] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    summary_path = latest_summary(trace_root, args.task_name, args.task_config, seed)
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    args = parse_args()
    if not args.ckpt_setting:
        args.ckpt_setting = f"{args.task_config}-{args.expert_data_num}"
    seeds = load_dataset_seeds(args)
    train_indices, val_indices = split_indices(len(seeds), args.train_ratio, args.split_seed)

    selected: list[tuple[str, int, int]] = []
    for idx in train_indices[: args.train_count]:
        selected.append(("train", int(idx), int(seeds[int(idx)])))
    for idx in val_indices[: args.val_count]:
        selected.append(("val", int(idx), int(seeds[int(idx)])))

    out_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"{args.task_name}_{args.task_config}_{args.ckpt_setting}_{stamp}.json"

    rows = []
    for split, episode_index, seed in selected:
        summary = run_trace(args, seed)
        row = {
            "split": split,
            "episode_index": episode_index,
            "seed": seed,
            "success": bool(summary.get("final_eval_success")),
            "steps": int(summary.get("steps", 0)),
            "final_step": summary.get("final_step"),
            "final_left_gripper_cmd": summary.get("final_left_gripper_cmd"),
            "final_right_gripper_cmd": summary.get("final_right_gripper_cmd"),
            "summary_path": summary.get("summary_path"),
        }
        rows.append(row)
        summary_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print("[act-seed-matrix] row=" + json.dumps(row, ensure_ascii=True), flush=True)

    print(f"[act-seed-matrix] summary={summary_path}")
    print(json.dumps({"args": vars(args), "rows": rows}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
