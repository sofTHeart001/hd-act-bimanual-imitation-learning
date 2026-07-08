"""GPU-free tests for run_act_eval.py pure helpers.

The heavy ACT-in-process backend (sim/torch/RoboTwin) is smoke-tested on a GPU
separately; here we only cover the player-code-shadow resolver, which is pure
filesystem + sys.path manipulation (codex route① #3).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import run_act_eval  # noqa: E402


def test_shadow_player_act_noop_when_none():
    before = list(sys.path)
    run_act_eval.shadow_player_act(None)
    assert sys.path == before          # 无代码包 → 不动 sys.path
    run_act_eval.shadow_player_act("")  # 空串同样 no-op
    assert sys.path == before


def test_shadow_player_act_inserts_player_policy_first(tmp_path, monkeypatch):
    (tmp_path / "policy" / "ACT").mkdir(parents=True)
    (tmp_path / "policy" / "ACT" / "__init__.py").write_text("# player ACT")
    saved = list(sys.path)
    monkeypatch.setattr(sys, "path", list(saved))
    run_act_eval.shadow_player_act(str(tmp_path))
    # <player>/policy 必须在 sys.path[0](才能让 import ACT 解析到选手副本)
    assert sys.path[0] == str((tmp_path / "policy").resolve())


def test_shadow_player_act_fail_loud_when_no_policy_act(tmp_path):
    # 给了代码包但没有 policy/ACT/__init__.py → 必须 fail-loud,绝不静默回落官方 ACT
    (tmp_path / "policy").mkdir(parents=True)
    with pytest.raises(SystemExit, match="policy/ACT"):
        run_act_eval.shadow_player_act(str(tmp_path))
