"""Unit tests for the T4 graded stacking score (pure, no sim/GPU).

Covers: all 4 levels, anchor-C tightening, anti-gaming cap, gripper gate,
table_z_bias, order-independence across overlapping bands, and input validation.
See spec §3 (docs/superpowers/specs/2026-06-29-troncamp-act-4task-suite-design.md).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graded_score import graded_stack_score, ANCHOR_C, LAYER_HEIGHTS  # noqa: E402

CX, CY = ANCHOR_C
H1, H2, H3 = LAYER_HEIGHTS


def _score(bowls, bias=0.0, left=True, right=True):
    return graded_stack_score(bowls, table_z_bias=bias,
                              left_gripper_open=left, right_gripper_open=right)


def _stack_at(cx, cy, bias=0.0):
    """Three bowls perfectly stacked at (cx, cy) at the three layer heights."""
    return [(cx, cy, H1 + bias), (cx, cy, H2 + bias), (cx, cy, H3 + bias)]


def test_full_success_is_one():
    assert _score(_stack_at(CX, CY)) == 1.0


def test_two_layers_is_two_thirds():
    # base + middle at anchor; third bowl dumped elsewhere (off-anchor, table z)
    bowls = [(CX, CY, H1), (CX, CY, H2), (0.25, 0.1, H1)]
    assert _score(bowls) == 2.0 / 3.0


def test_one_layer_is_one_third():
    bowls = [(CX, CY, H1), (0.25, 0.1, H1), (-0.25, 0.1, H1)]
    assert _score(bowls) == 1.0 / 3.0


def test_nothing_at_anchor_is_zero():
    bowls = [(0.25, 0.1, H1), (0.25, 0.1, H2), (0.25, 0.1, H3)]
    assert _score(bowls) == 0.0


def test_anchor_tightening_perfect_stack_offset_scores_zero():
    # a perfect 3-bowl stack, but NOT at the anchor C -> 0 (tightened to C)
    assert _score(_stack_at(0.2, 0.0)) == 0.0


def test_anti_gaming_two_bowls_on_table_at_anchor_caps_at_one_third():
    # shove bowls to C on the table without stacking: only layer 1 credited
    bowls = [(CX, CY, H1), (CX, CY, H1), (0.25, 0.1, H1)]
    assert _score(bowls) == 1.0 / 3.0


def test_gripper_gate_closed_gripper_caps_at_one_third():
    # 第 2 层起要求双夹爪张开(held ≠ placed):夹爪未开时,即使三碗都在层高,
    # 也只给第 1 层(桌面高度,等价于已接受的 shoved-to-C 情形)。
    bowls = _stack_at(CX, CY)
    assert _score(bowls, right=False) == 1.0 / 3.0
    assert _score(bowls, left=False) == 1.0 / 3.0


def test_gripper_gate_held_bowl_at_layer2_height_not_scored():
    # 反刷分:一碗放好、另一碗被夹在第 2 层高度悬停 → 夹爪闭合,只记第 1 层。
    bowls = [(CX, CY, H1), (CX, CY, H2), (0.25, 0.1, H1)]
    assert _score(bowls) == 2.0 / 3.0                 # 真放好(夹爪开)→ 2/3
    assert _score(bowls, left=False) == 1.0 / 3.0     # 夹着悬停 → 1/3


def test_table_z_bias_is_applied():
    assert _score(_stack_at(CX, CY, bias=0.01), bias=0.01) == 1.0


def test_score_is_order_independent_with_overlapping_bands():
    # middle bowl at 0.755 sits in the [0.75, 0.76] overlap of the base/middle
    # height bands; a real 3-stack must score 1.0 regardless of input order
    # (first-match greedy used to return 1/3 for some orderings).
    perm_a = [(CX, CY, 0.755), (CX, CY, 0.740), (CX, CY, 0.810)]
    perm_b = [(CX, CY, 0.810), (CX, CY, 0.755), (CX, CY, 0.740)]
    assert _score(perm_a) == 1.0
    assert _score(perm_b) == 1.0


def test_duplicate_height_bowls_do_not_double_count():
    # two bowls at the same z in the band overlap must NOT count as base+middle
    # (no real vertical separation): base credited, middle rejected -> 1/3.
    bowls = [(CX, CY, 0.755), (CX, CY, 0.755), (CX, CY, 0.810)]
    assert _score(bowls) == 1.0 / 3.0


def test_grippers_are_required_keyword_args():
    with pytest.raises(TypeError):
        graded_stack_score(_stack_at(CX, CY))  # missing required gripper flags


def test_requires_exactly_three_bowls():
    with pytest.raises(ValueError):
        _score([(CX, CY, H1), (CX, CY, H2)])


def test_rejects_position_without_z():
    with pytest.raises(ValueError):
        _score([(CX, CY), (CX, CY, H2), (CX, CY, H3)])
