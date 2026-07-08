#!/usr/bin/env python3
"""Basic local checks before the official RoboTwin package is available."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "external" / "robotwin_local"
ACT = ROBOTWIN / "policy" / "ACT"


def ok(message: str) -> None:
    print(f"[ok] {message}")


def warn(message: str) -> None:
    print(f"[warn] {message}")


def check_path(path: Path) -> bool:
    if path.exists():
        ok(f"存在: {path.relative_to(ROOT)}")
        return True
    warn(f"缺少: {path.relative_to(ROOT)}")
    return False


def command_output(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return 127, ""
    return proc.returncode, proc.stdout.strip()


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    failed = False

    print(f"[info] workspace: {ROOT}")
    print(f"[info] python: {sys.version.split()[0]} ({sys.executable})")
    print(f"[info] conda env: {os.environ.get('CONDA_DEFAULT_ENV', '<not active>')}")
    if sys.version_info[:2] == (3, 10):
        ok("Python 版本符合参赛文档要求: 3.10")
    else:
        warn("参赛文档要求 Python 3.10；当前解释器不是 3.10。")

    for cmd in ("conda", "ffmpeg", "python3"):
        resolved = shutil.which(cmd)
        if resolved:
            ok(f"{cmd}: {resolved}")
        else:
            warn(f"缺少命令: {cmd}")
            failed = True

    code, output = command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if code == 0 and output:
        ok(f"NVIDIA: {output.splitlines()[0]}")
    else:
        warn("nvidia-smi 不可用；采集/训练/评测需要可用 NVIDIA 驱动。")

    required = [
        ROBOTWIN,
        ROBOTWIN / "script" / "_install.sh",
        ROBOTWIN / "task_config" / "adjust_bottle_200ep.yml",
        ACT / "process_data.sh",
        ACT / "train.sh",
        ACT / "eval.sh",
    ]
    for path in required:
        if not check_path(path):
            failed = True

    code, output = command_output(["grep", "-R", "--include=*.yml", "-n", "__KIT_ROOT__", str(ROOT)])
    if code == 0 and output:
        warn("__KIT_ROOT__ 占位符仍存在，请执行 make restore-paths")
    else:
        ok("__KIT_ROOT__ yml 占位符未发现")

    return 1 if strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
