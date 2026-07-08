#!/usr/bin/env python3
"""Patch the project-local RoboTwin ACT runtime for Tron2 state/action dims.

This script intentionally writes only under <robotwin>/policy/ACT and refuses
the shared upstream external/robotwin checkout.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


class PatchError(RuntimeError):
    pass


def subrepo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def default_robotwin_root() -> Path:
    return subrepo_root() / "external" / "robotwin_local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root",
        default=os.environ.get("TRON2_ROBOTWIN_DIR", str(default_robotwin_root())),
        help="Project-local RoboTwin runtime root. Defaults to TRON2_ROBOTWIN_DIR or external/robotwin_local.",
    )
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_local_runtime(robotwin_root: Path) -> Path:
    root = resolve_path(robotwin_root)
    shared = resolve_path(subrepo_root() / "external" / "robotwin")

    if root == shared or shared in root.parents:
        raise PatchError(f"Refusing to patch shared upstream RoboTwin checkout: {root}")
    if root.name == "robotwin" and root.parent.name == "external":
        raise PatchError(f"Refusing suspicious shared RoboTwin path: {root}")

    act_dir = root / "policy" / "ACT"
    if not act_dir.is_dir():
        raise PatchError(f"ACT runtime directory not found: {act_dir}")
    act_real = act_dir.resolve(strict=True)
    if act_real == shared or shared in act_real.parents:
        raise PatchError(f"Refusing ACT runtime that resolves into shared upstream RoboTwin: {act_real}")
    return act_dir


def rewrite_python(path: Path, label: str, patches: list[tuple[str, str, str]]) -> list[str]:
    text = path.read_text()
    new_text = text
    status: list[str] = []

    for name, already_re, patch_re in patches:
        if re.search(already_re, new_text, flags=re.MULTILINE):
            status.append(f"{label}: {name} already patched")
            continue

        replacement = patches_replacements[name]
        new_text, count = re.subn(patch_re, replacement, new_text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise PatchError(f"Could not find expected anchor for {name} in {path}")
        status.append(f"{label}: patched {name}")

    if new_text != text:
        path.write_text(new_text)
    return status


patches_replacements = {
    "imitate state_dim": r'\1state_dim = args["state_dim"]\n',
    "imitate optional sim_env": (
        "try:\n"
        "    from sim_env import BOX_POSE\n"
        "except ModuleNotFoundError:\n"
        "    # Offline ACT training does not need RoboTwin's dm_control sim_env.\n"
        "    BOX_POSE = [None]\n"
    ),
    "act_policy ckpt_name": (
        r'\1ckpt_name = args_override.get("ckpt_name", "policy_last.ckpt")\n'
        r"\1ckpt_path = os.path.join(ckpt_dir, ckpt_name)\n"
    ),
    "act_policy camera_names": (
        r"\1self.state_dim = RoboTwin_Config.action_dim  # Standard joint dimension for bimanual robot\n"
        r'\1self.camera_names = args_override.get("camera_names", ["cam_high", "cam_right_wrist", "cam_left_wrist"])\n'
    ),
    "act_policy eval camera order": (
        r"\1# Prepare images following imitate_episodes.py pattern\n"
        r"\1# Stack images in the same semantic order used during training.\n"
        r"\1camera_key_map = {\n"
        r'\1    "cam_high": "head_cam",\n'
        r'\1    "cam_right_wrist": "right_cam",\n'
        r'\1    "cam_left_wrist": "left_cam",\n'
        r'\1    "head_cam": "head_cam",\n'
        r'\1    "right_cam": "right_cam",\n'
        r'\1    "left_cam": "left_cam",\n'
        r"\1}\n"
        r"\1curr_images = []\n"
        r"\1for cam_name in self.camera_names:\n"
        r"\1    curr_images.append(obs[camera_key_map.get(cam_name, cam_name)])\n"
    ),
    "eval configurable test_num": r'\1test_num = int(usr_args.get("test_num", 100))\n',
    "eval runtime overrides": (
        r'\1args["ckpt_setting"] = ckpt_setting\n\n'
        r'\1for key in ("eval_video_log", "render_freq", "clear_cache_freq"):\n'
        r"\1    if key in usr_args:\n"
        r"\1        args[key] = usr_args[key]\n"
    ),
    "eval configurable expert_check": r'\1expert_check = bool(usr_args.get("expert_check", True))\n',
    "eval progress loop": (
        r'\1while succ_seed < test_num:\n'
        r'\1    print(f"[eval-policy] loop now_seed={now_seed} succ_seed={succ_seed}/{test_num}", flush=True)\n'
    ),
    "eval progress expert": (
        r'\1print(f"[eval-policy] expert setup seed={now_seed}", flush=True)\n'
        r"\1TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        r'\1print(f"[eval-policy] expert play_once seed={now_seed}", flush=True)\n'
        r"\1episode_info = TASK_ENV.play_once()\n"
        r'\1print(f"[eval-policy] expert done seed={now_seed} plan_success={TASK_ENV.plan_success}", flush=True)\n'
    ),
    "eval no expert instruction": (
        r"\1TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        r"\1if expert_check:\n"
        r'\1    episode_info_list = [episode_info["info"]]\n'
        r'\1    results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)\n'
        r"\1    instruction = np.random.choice(results[0][instruction_type])\n"
        r"\1else:\n"
        r'\1    instruction = ""\n'
        r'\1    print(f"[eval-policy] expert_check disabled; using empty instruction for seed={now_seed}", flush=True)\n'
    ),
    "utils sim action start": (
        r"\1else:\n"
        r"\1    # Tron2 processed data stores /action[t] as the next saved qpos.\n"
        r'\1    action = root["/action"][start_ts:]\n'
        r"\1    action_len = episode_len - start_ts\n"
    ),
    "detr build state_dim": r"\1state_dim = args.state_dim\n",
    "detr build_cnnmlp state_dim": r"\1state_dim = args.state_dim\n",
}


def patch_imitate(act_dir: Path) -> list[str]:
    return rewrite_python(
        act_dir / "imitate_episodes.py",
        "imitate_episodes.py",
        [
            (
                "imitate state_dim",
                r'state_dim\s*=\s*args\["state_dim"\]',
                r"(# fixed parameters\n\s*)state_dim\s*=\s*14\s*(?:#.*)?\n",
            ),
            (
                "imitate optional sim_env",
                r"try:\n\s*from sim_env import BOX_POSE\nexcept ModuleNotFoundError:\n\s*# Offline ACT training does not need RoboTwin's dm_control sim_env\.\n\s*BOX_POSE = \[None\]",
                r"from sim_env import BOX_POSE\n",
            ),
        ],
    )


def patch_detr_vae(act_dir: Path) -> list[str]:
    return rewrite_python(
        act_dir / "detr" / "models" / "detr_vae.py",
        "detr/models/detr_vae.py",
        [
            (
                "detr build state_dim",
                r"def build\(args\):\n\s*state_dim\s*=\s*args\.state_dim",
                r"(def build\(args\):\n\s*)state_dim\s*=\s*14\s*(?:#.*)?\n",
            ),
            (
                "detr build_cnnmlp state_dim",
                r"def build_cnnmlp\(args\):\n\s*state_dim\s*=\s*args\.state_dim",
                r"(def build_cnnmlp\(args\):\n\s*)state_dim\s*=\s*16\s*(?:#.*)?\n",
            ),
        ],
    )


def patch_utils_alignment(act_dir: Path) -> list[str]:
    return rewrite_python(
        act_dir / "utils.py",
        "utils.py",
        [
            (
                "utils sim action start",
                r"Tron2 processed data stores /action\[t\] as the next saved qpos\.\n\s*action\s*=\s*root\[\"/action\"\]\[start_ts:\]\n\s*action_len\s*=\s*episode_len\s*-\s*start_ts",
                r'(\s*)else:\n\s*action\s*=\s*root\["/action"\]\[max\(0,\s*start_ts\s*-\s*1\):\]\s*# hack, to make timesteps more aligned\n\s*action_len\s*=\s*episode_len\s*-\s*max\(0,\s*start_ts\s*-\s*1\)\s*# hack, to make timesteps more aligned\n',
            ),
        ],
    )


def patch_act_policy(act_dir: Path) -> list[str]:
    return rewrite_python(
        act_dir / "act_policy.py",
        "act_policy.py",
        [
            (
                "act_policy ckpt_name",
                r'ckpt_name\s*=\s*args_override\.get\("ckpt_name",\s*"policy_last\.ckpt"\)\n\s*ckpt_path\s*=\s*os\.path\.join\(ckpt_dir,\s*ckpt_name\)',
                r'(\s*)ckpt_path\s*=\s*os\.path\.join\(ckpt_dir,\s*"policy_last\.ckpt"\)\n',
            ),
            (
                "act_policy camera_names",
                r'self\.camera_names\s*=\s*args_override\.get\("camera_names",\s*\["cam_high",\s*"cam_right_wrist",\s*"cam_left_wrist"\]\)',
                r"(\s*)self\.state_dim\s*=\s*RoboTwin_Config\.action_dim\s*#.*\n",
            ),
            (
                "act_policy eval camera order",
                r'camera_key_map\s*=\s*\{\n\s*"cam_high":\s*"head_cam",\n\s*"cam_right_wrist":\s*"right_cam",\n\s*"cam_left_wrist":\s*"left_cam",\n\s*"head_cam":\s*"head_cam",\n\s*"right_cam":\s*"right_cam",\n\s*"left_cam":\s*"left_cam",\n\s*\}\n\s*curr_images\s*=\s*\[\]\n\s*for cam_name in self\.camera_names:\n\s*curr_images\.append\(obs\[camera_key_map\.get\(cam_name,\s*cam_name\)\]\)',
                r'([ \t]*)# Prepare images following imitate_episodes\.py pattern\n[ \t]*# Stack images from all cameras\n[ \t]*curr_images = \[\]\n[ \t]*camera_names = \["head_cam", "left_cam", "right_cam"\]\n[ \t]*for cam_name in camera_names:\n[ \t]*curr_images\.append\(obs\[cam_name\]\)',
            ),
        ],
    )


def patch_eval_policy(act_dir: Path) -> list[str]:
    robotwin_root = act_dir.parents[1]
    return rewrite_python(
        robotwin_root / "script" / "eval_policy.py",
        "script/eval_policy.py",
        [
            (
                "eval configurable test_num",
                r'test_num\s*=\s*int\(usr_args\.get\("test_num",\s*100\)\)',
                r"(\s*)test_num\s*=\s*100\n",
            ),
            (
                "eval runtime overrides",
                r'args\["ckpt_setting"\]\s*=\s*ckpt_setting\n\n\s*for key in \("eval_video_log", "render_freq", "clear_cache_freq"\):\n\s*if key in usr_args:\n\s*args\[key\] = usr_args\[key\]',
                r'(\s*)args\["ckpt_setting"\]\s*=\s*ckpt_setting\n',
            ),
            (
                "eval configurable expert_check",
                r'expert_check\s*=\s*bool\(usr_args\.get\("expert_check",\s*True\)\)',
                r"(\s*)expert_check\s*=\s*True\n",
            ),
            (
                "eval progress loop",
                r'while succ_seed < test_num:\n\s*print\(f"\[eval-policy\] loop now_seed=\{now_seed\} succ_seed=\{succ_seed\}/\{test_num\}", flush=True\)',
                r"(\s*)while succ_seed < test_num:\n",
            ),
            (
                "eval progress expert",
                r'print\(f"\[eval-policy\] expert setup seed=\{now_seed\}", flush=True\)\n\s*TASK_ENV\.setup_demo\(now_ep_num=now_id, seed=now_seed, is_test=True, \*\*args\)\n\s*print\(f"\[eval-policy\] expert play_once seed=\{now_seed\}", flush=True\)\n\s*episode_info = TASK_ENV\.play_once\(\)\n\s*print\(f"\[eval-policy\] expert done seed=\{now_seed\} plan_success=\{TASK_ENV\.plan_success\}", flush=True\)',
                r"(\s*)TASK_ENV\.setup_demo\(now_ep_num=now_id, seed=now_seed, is_test=True, \*\*args\)\n\s*episode_info = TASK_ENV\.play_once\(\)\n",
            ),
            (
                "eval no expert instruction",
                r'TASK_ENV\.setup_demo\(now_ep_num=now_id, seed=now_seed, is_test=True, \*\*args\)\n\s*if expert_check:\n\s*episode_info_list = \[episode_info\["info"\]\]\n\s*results = generate_episode_descriptions\(args\["task_name"\], episode_info_list, test_num\)\n\s*instruction = np\.random\.choice\(results\[0\]\[instruction_type\]\)\n\s*else:\n\s*instruction = ""\n\s*print\(f"\[eval-policy\] expert_check disabled; using empty instruction for seed=\{now_seed\}", flush=True\)',
                r'(\s*)TASK_ENV\.setup_demo\(now_ep_num=now_id, seed=now_seed, is_test=True, \*\*args\)\n\s*episode_info_list = \[episode_info\["info"\]\]\n\s*results = generate_episode_descriptions\(args\["task_name"\], episode_info_list, test_num\)\n\s*instruction = np\.random\.choice\(results\[0\]\[instruction_type\]\)\n',
            ),
        ],
    )


def patch_deploy_config(act_dir: Path, state_dim: int) -> list[str]:
    path = act_dir / "deploy_policy.yml"
    if not path.exists():
        return [f"{path}: skipped deploy action_dim patch; file not found"]

    text = path.read_text()
    status: list[str] = []

    match = re.search(r"(?m)^(action_dim:\s*)(\d+)(\s*)$", text)
    if not match:
        status.append(f"{path}: skipped deploy action_dim patch; no scalar action_dim line found")
    else:
        current = int(match.group(2))
        if current == state_dim:
            status.append(f"{path.name}: action_dim already {state_dim}")
        else:
            text = text[:match.start(2)] + str(state_dim) + text[match.end(2):]
            status.append(f"{path.name}: action_dim {current} -> {state_dim}")

    state_match = re.search(r"(?m)^(state_dim:\s*)(\d+)(\s*)$", text)
    if state_match:
        current = int(state_match.group(2))
        if current == state_dim:
            status.append(f"{path.name}: state_dim already {state_dim}")
        else:
            text = text[:state_match.start(2)] + str(state_dim) + text[state_match.end(2):]
            status.append(f"{path.name}: state_dim {current} -> {state_dim}")
    elif match:
        insert_at = match.end(0)
        text = text[:insert_at] + f"\nstate_dim: {state_dim}" + text[insert_at:]
        status.append(f"{path.name}: inserted state_dim {state_dim}")
    else:
        text = text.rstrip() + f"\nstate_dim: {state_dim}\n"
        status.append(f"{path.name}: appended state_dim {state_dim}")

    path.write_text(text)
    return status


def main() -> int:
    args = parse_args()
    try:
        act_dir = ensure_local_runtime(Path(args.robotwin_root))
        actions = []
        if args.dry_run:
            print(f"[dry-run] would patch ACT runtime under: {act_dir}")
            return 0
        actions.extend(patch_imitate(act_dir))
        actions.extend(patch_utils_alignment(act_dir))
        actions.extend(patch_detr_vae(act_dir))
        actions.extend(patch_act_policy(act_dir))
        actions.extend(patch_eval_policy(act_dir))
        actions.extend(patch_deploy_config(act_dir, args.state_dim))
    except PatchError as exc:
        print(f"[patch-act-runtime] ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[patch-act-runtime] ACT runtime: {act_dir}")
    for action in actions:
        print(f"[patch-act-runtime] {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
