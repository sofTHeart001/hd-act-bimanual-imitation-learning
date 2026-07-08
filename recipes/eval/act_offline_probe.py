#!/usr/bin/env python3
"""Probe ACT checkpoint action error on processed training episodes."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--target-start",
        choices=("current", "previous"),
        default=os.environ.get("TRON2_ACT_TARGET_START", "current"),
        help="Use current for the align-next dataloader patch, previous for the original RoboTwin hack.",
    )
    return parser.parse_args()


def load_sim_config(act_dir: Path, task_name: str, task_config: str, expert_data_num: int) -> tuple[str, Path, list[str]]:
    sim_key = f"sim-{task_name}-{task_config}-{expert_data_num}"
    with (act_dir / "SIM_TASK_CONFIGS.json").open() as f:
        configs = json.load(f)
    cfg = configs[sim_key]
    dataset_dir = Path(cfg["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = act_dir / dataset_dir
    return sim_key, dataset_dir, list(cfg["camera_names"])


def image_stack(root, camera_names: list[str], frame: int):
    images = []
    for camera_name in camera_names:
        image = root[f"/observations/images/{camera_name}"][frame]
        images.append(np.moveaxis(image, -1, 0))
    return np.stack(images, axis=0).astype(np.float32) / 255.0


def mean_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    robotwin_root = Path(args.robotwin_root).expanduser().resolve(strict=False)
    act_dir = robotwin_root / "policy" / "ACT"
    ckpt_setting = args.ckpt_setting or f"{args.task_config}-{args.expert_data_num}"
    ckpt_dir = act_dir / "act_ckpt" / f"tron2-{args.task_name}" / ckpt_setting
    ckpt_path = ckpt_dir / args.ckpt_name
    stats_path = ckpt_dir / "dataset_stats.pkl"

    repo = subrepo_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(act_dir))

    from recipes.train.patch_act_runtime import (
        ensure_local_runtime,
        patch_act_policy,
        patch_deploy_config,
        patch_detr_vae,
        patch_eval_policy,
        patch_imitate,
    )

    runtime_act_dir = ensure_local_runtime(robotwin_root)
    patch_imitate(runtime_act_dir)
    patch_detr_vae(runtime_act_dir)
    patch_act_policy(runtime_act_dir)
    patch_eval_policy(runtime_act_dir)
    patch_deploy_config(runtime_act_dir, args.state_dim)

    import h5py
    import torch
    from act_policy import ACTPolicy

    sim_key, dataset_dir, camera_names = load_sim_config(act_dir, args.task_name, args.task_config, args.expert_data_num)
    with stats_path.open("rb") as f:
        stats = pickle.load(f)

    with (act_dir / "deploy_policy.yml").open() as f:
        policy_args = yaml.safe_load(f)
    policy_args.update(
        {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "ckpt_setting": ckpt_setting,
            "ckpt_dir": str(ckpt_dir.relative_to(robotwin_root)),
            "ckpt_name": args.ckpt_name,
            "state_dim": args.state_dim,
            "action_dim": args.state_dim,
            "temporal_agg": True,
            "camera_names": camera_names,
            "left_arm_dim": 7,
            "right_arm_dim": 7,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }
    )

    policy = ACTPolicy(policy_args, Namespace(**policy_args))
    state_dict = torch.load(ckpt_path, map_location=policy_args["device"])
    policy.load_state_dict(state_dict)
    policy.to(policy_args["device"])
    policy.eval()

    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)

    train_target_l1 = []
    pred0_to_current_qpos_l1 = []
    pred0_to_next_action_l1 = []
    target0_to_current_qpos_l1 = []
    target0_to_next_action_l1 = []

    episode_count = min(args.episodes, args.expert_data_num)
    with torch.inference_mode():
        for episode_idx in range(episode_count):
            episode_path = dataset_dir / f"episode_{episode_idx}.hdf5"
            with h5py.File(episode_path, "r") as root:
                qpos_arr = root["/observations/qpos"]
                action_arr = root["/action"]
                frame_count = min(qpos_arr.shape[0], action_arr.shape[0])
                if frame_count <= 1:
                    continue
                frame_ids = np.linspace(0, frame_count - 1, min(args.frames_per_episode, frame_count), dtype=int)
                for frame in frame_ids:
                    qpos = np.asarray(qpos_arr[frame], dtype=np.float32)
                    images = image_stack(root, camera_names, int(frame))

                    qpos_norm = (qpos - qpos_mean) / qpos_std
                    qpos_tensor = torch.from_numpy(qpos_norm).float().to(policy_args["device"]).unsqueeze(0)
                    image_tensor = torch.from_numpy(images).float().to(policy_args["device"]).unsqueeze(0)

                    raw = policy(qpos_tensor, image_tensor).squeeze(0).detach().cpu().numpy()
                    pred = raw * action_std + action_mean

                    if args.target_start == "previous":
                        target_start = max(0, int(frame) - 1)
                    else:
                        target_start = int(frame)
                    target_chunk = np.asarray(action_arr[target_start:target_start + pred.shape[0]], dtype=np.float32)
                    if target_chunk.shape[0] == 0:
                        continue
                    pred_chunk = pred[: target_chunk.shape[0]]
                    train_target_l1.append(mean_l1(pred_chunk, target_chunk))

                    pred0 = pred[0]
                    target0 = target_chunk[0]
                    next_action = np.asarray(action_arr[min(int(frame), action_arr.shape[0] - 1)], dtype=np.float32)
                    pred0_to_current_qpos_l1.append(mean_l1(pred0, qpos))
                    pred0_to_next_action_l1.append(mean_l1(pred0, next_action))
                    target0_to_current_qpos_l1.append(mean_l1(target0, qpos))
                    target0_to_next_action_l1.append(mean_l1(target0, next_action))

    def summarize(values: list[float]) -> str:
        arr = np.asarray(values, dtype=np.float32)
        return f"mean={arr.mean():.6f} median={np.median(arr):.6f} p90={np.quantile(arr, 0.9):.6f} n={arr.size}"

    print(f"[act-offline-probe] sim_key={sim_key}")
    print(f"[act-offline-probe] ckpt={ckpt_path}")
    print(f"[act-offline-probe] dataset={dataset_dir}")
    print(f"[act-offline-probe] cameras={camera_names}")
    print(f"[act-offline-probe] target_start={args.target_start}")
    print(f"[act-offline-probe] train_target_chunk_l1 {summarize(train_target_l1)}")
    print(f"[act-offline-probe] pred0_to_current_qpos_l1 {summarize(pred0_to_current_qpos_l1)}")
    print(f"[act-offline-probe] pred0_to_next_action_l1 {summarize(pred0_to_next_action_l1)}")
    print(f"[act-offline-probe] target0_to_current_qpos_l1 {summarize(target0_to_current_qpos_l1)}")
    print(f"[act-offline-probe] target0_to_next_action_l1 {summarize(target0_to_next_action_l1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
