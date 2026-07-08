"""Resolve the project-local RoboTwin runtime checkout."""

from __future__ import annotations

import os


def get_robotwin_dir(subrepo: str) -> str:
    """Return the writable RoboTwin checkout for Tron2 rollout scripts."""

    robotwin = os.environ.get("TRON2_ROBOTWIN_DIR")
    if robotwin is None:
        robotwin = os.path.join(subrepo, "external", "robotwin_local")
    robotwin = os.path.abspath(robotwin)

    if not os.path.isdir(robotwin):
        raise RuntimeError(
            "RoboTwin writable runtime checkout is missing: "
            f"{robotwin}. Run recipes/rollout/bootstrap_robotwin_local.sh first, "
            "or set TRON2_ROBOTWIN_DIR to a project-local copy."
        )
    shared = os.path.abspath(os.path.join(subrepo, "external", "robotwin"))
    if os.path.isdir(shared) and os.path.realpath(robotwin) == os.path.realpath(shared):
        raise RuntimeError(
            "Refusing to use shared external/robotwin as a writable runtime. "
            "Run recipes/rollout/bootstrap_robotwin_local.sh and use "
            "external/robotwin_local instead."
        )
    return robotwin
