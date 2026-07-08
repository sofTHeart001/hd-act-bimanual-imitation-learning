#!/usr/bin/env python3
"""Validate processed ACT qpos/action dimensions before training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


EXPECTED_CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]


def subrepo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_robotwin_root() -> Path:
    return subrepo_root() / "external" / "robotwin_local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root",
        default=os.environ.get("TRON2_ROBOTWIN_DIR", str(default_robotwin_root())),
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--expert-data-num", required=True, type=int)
    parser.add_argument("--state-dim", required=True, type=int)
    return parser.parse_args()


def load_dataset_dir(
    act_dir: Path, task_name: str, task_config: str, expert_data_num: int
) -> tuple[str, Path, list[str]]:
    sim_key = f"sim-{task_name}-{task_config}-{expert_data_num}"
    config_path = act_dir / "SIM_TASK_CONFIGS.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing {config_path}; run ACT process_data.py before training")

    with config_path.open() as f:
        configs = json.load(f)

    if sim_key not in configs:
        raise RuntimeError(f"Missing {sim_key!r} in {config_path}; run ACT process_data.py for this dataset")

    dataset_value = configs[sim_key].get("dataset_dir")
    if not dataset_value:
        raise RuntimeError(f"{sim_key!r} has no dataset_dir in {config_path}")

    camera_names = configs[sim_key].get("camera_names")
    if not isinstance(camera_names, list) or not camera_names:
        raise RuntimeError(f"{sim_key!r} has no non-empty camera_names list in {config_path}")
    if not all(isinstance(name, str) and name for name in camera_names):
        raise RuntimeError(f"{sim_key!r} camera_names must be non-empty strings in {config_path}")
    if camera_names != EXPECTED_CAMERA_NAMES:
        raise RuntimeError(
            f"{sim_key!r} camera_names {camera_names} != expected Tron2 ACT order {EXPECTED_CAMERA_NAMES}"
        )

    config_num_episodes = configs[sim_key].get("num_episodes")
    if config_num_episodes != expert_data_num:
        raise RuntimeError(
            f"{sim_key!r} num_episodes {config_num_episodes} != expected {expert_data_num}"
        )

    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = act_dir / dataset_path
    return sim_key, dataset_path.resolve(strict=False), camera_names


def validate_episode(path: Path, state_dim: int, camera_names: list[str]) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on server env
        raise RuntimeError("h5py is required for HDF5 preflight") from exc

    with h5py.File(path, "r") as root:
        missing = [key for key in ("/observations/qpos", "/action") if key not in root]
        if missing:
            raise RuntimeError(f"{path} missing datasets: {', '.join(missing)}")

        qpos_shape = root["/observations/qpos"].shape
        action_shape = root["/action"].shape
        for camera_name in camera_names:
            image_key = f"/observations/images/{camera_name}"
            if image_key not in root:
                raise RuntimeError(f"{path} missing camera dataset: {image_key}")
            image = root[image_key]
            image_shape = image.shape
            if len(image_shape) != 4:
                raise RuntimeError(f"{path} {image_key} must be rank-4 NHWC, got {image_shape}")
            if image_shape[0] != qpos_shape[0]:
                raise RuntimeError(
                    f"{path} {image_key} length {image_shape[0]} != qpos length {qpos_shape[0]}"
                )
            if image_shape[-1] != 3:
                raise RuntimeError(f"{path} {image_key} must have 3 color channels, got {image_shape}")
            if image.dtype.kind != "u" or image.dtype.itemsize != 1:
                raise RuntimeError(f"{path} {image_key} must be uint8, got {image.dtype}")

    if len(qpos_shape) != 2:
        raise RuntimeError(f"{path} /observations/qpos must be rank-2, got {qpos_shape}")
    if len(action_shape) != 2:
        raise RuntimeError(f"{path} /action must be rank-2, got {action_shape}")
    if qpos_shape[1] != state_dim:
        raise RuntimeError(f"{path} qpos dim {qpos_shape[1]} != expected state_dim {state_dim}")
    if action_shape[1] != state_dim:
        raise RuntimeError(f"{path} action dim {action_shape[1]} != expected state_dim {state_dim}")
    if qpos_shape[0] <= 0 or action_shape[0] <= 0:
        raise RuntimeError(f"{path} has empty qpos/action sequence: {qpos_shape}, {action_shape}")
    if qpos_shape[0] != action_shape[0]:
        raise RuntimeError(f"{path} qpos/action length mismatch: {qpos_shape}, {action_shape}")


def main() -> int:
    args = parse_args()
    robotwin_root = Path(args.robotwin_root).expanduser().resolve(strict=False)
    act_dir = robotwin_root / "policy" / "ACT"
    if not act_dir.is_dir():
        print(f"[validate-act-shapes] ERROR: ACT runtime directory not found: {act_dir}", file=sys.stderr)
        return 2

    try:
        sim_key, dataset_dir, camera_names = load_dataset_dir(
            act_dir, args.task_name, args.task_config, args.expert_data_num
        )
        if not dataset_dir.is_dir():
            raise RuntimeError(f"Processed ACT dataset directory not found: {dataset_dir}")

        missing_files = []
        for episode_idx in range(args.expert_data_num):
            episode_path = dataset_dir / f"episode_{episode_idx}.hdf5"
            if not episode_path.is_file():
                missing_files.append(str(episode_path))
                continue
            validate_episode(episode_path, args.state_dim, camera_names)

        if missing_files:
            raise RuntimeError("Missing processed ACT episode files:\n" + "\n".join(missing_files[:10]))
    except RuntimeError as exc:
        print(f"[validate-act-shapes] ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"[validate-act-shapes] {sim_key}: validated {args.expert_data_num} episodes "
        f"with qpos/action dim {args.state_dim} and cameras {camera_names} in {dataset_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
