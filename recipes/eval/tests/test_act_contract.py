"""Tests for the ACT eval -> result contract producer (GPU-free).

Everything here runs against a FAKE rollout backend: the contract assembly, the
once-per-episode graded call, and the producer-side fail-loud guards are exercised
with synthetic final states, no sim/GPU. The real ACT-in-process backend (and the
per-STEP guard that graded_stack_score is never called inside the env step loop /
check_success) is wired + smoke-tested separately — it needs a GPU.
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from act_contract import (  # noqa: E402
    EpisodeOutcome,
    GRADED_TRACK,
    TASK_BY_TRACK,
    run_contract,
    write_result,
)
from graded_score import ANCHOR_C, LAYER_HEIGHTS  # noqa: E402

CX, CY = ANCHOR_C
H1, H2, H3 = LAYER_HEIGHTS


def _seeds(n):
    return [{"episode_seed": s} for s in range(n)]


def _full_stack_outcome(success=True):
    return EpisodeOutcome(
        success=success,
        final_bowls=[(CX, CY, H1), (CX, CY, H2), (CX, CY, H3)],
        left_gripper_open=True,
        right_gripper_open=True,
        table_z_bias=0.0,
    )


def _binary_backend(success_pattern):
    """Backend whose i-th seed (per repeat) succeeds per `success_pattern`."""

    def backend(episode_seed, repeat_idx, task):
        return EpisodeOutcome(success=success_pattern[episode_seed])

    return backend


# --- binary tracks (T1-T3): pure SR, no graded key ---------------------------


def test_t1_pure_sr_no_graded_key():
    backend = _binary_backend({0: True, 1: False, 2: True, 3: False})
    res = run_contract(_seeds(4), "T1", 1, backend)
    assert res["sr"] == 0.5
    assert res["track"] == "T1"
    assert res["n_episodes"] == 4
    assert res["n_repeats"] == 1
    assert "graded" not in res  # T1-T3 are pure-SR threshold gates


def test_binary_grader_never_called_for_t2():
    calls = {"n": 0}

    def spy_grader(*a, **k):
        calls["n"] += 1
        return 1.0

    backend = _binary_backend({0: True, 1: True})
    run_contract(_seeds(2), "T2", 1, backend, grader=spy_grader)
    assert calls["n"] == 0  # gates never grade


# --- T4 main board: float graded mean ----------------------------------------


def test_t4_emits_float_graded_mean():
    # two episodes: a perfect stack (1.0) and a base-only-at-anchor (1/3)
    outcomes = [
        _full_stack_outcome(success=True),
        EpisodeOutcome(
            success=False,
            final_bowls=[(CX, CY, H1), (0.25, 0.1, H1), (-0.25, 0.1, H1)],
            left_gripper_open=True,
            right_gripper_open=True,
        ),
    ]

    def backend(episode_seed, repeat_idx, task):
        return outcomes[episode_seed]

    res = run_contract(_seeds(2), "T4", 1, backend)
    assert res["track"] == GRADED_TRACK == "T4"
    assert isinstance(res["graded"], float)
    assert res["graded"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    # binary SR is still reported alongside the graded mean
    assert res["sr"] == 0.5


def test_t4_graded_is_real_float_not_bool():
    # guards against the retired pi05 contract where `graded` was a bool flag;
    # scoring_io.coerce_graded fail-closes on bools.
    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    res = run_contract(_seeds(3), "T4", 1, backend)
    assert res["graded"] == 1.0
    assert not isinstance(res["graded"], bool)


# --- grading is the ORCHESTRATOR's job, decoupled from rollout step count -----


def test_grader_called_exactly_once_per_episode():
    # The orchestrator calls the grader once per episode at episode END. The backend
    # returns raw final STATE (not a score) and cannot grade — that structural split
    # is what prevents the known 4M-call per-step averaging bug. A real backend that
    # (wrongly) graded inside its step loop is a Phase-2 / GPU concern, not testable
    # here; this pins the contract-layer invariant: calls == episode count.
    calls = {"n": 0}

    def spy_grader(*a, **k):
        calls["n"] += 1
        return 1.0

    n_seeds, repeats = 5, 3
    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    run_contract(_seeds(n_seeds), "T4", repeats, backend, grader=spy_grader)
    assert calls["n"] == n_seeds * repeats


def test_grader_calls_independent_of_internal_rollout_steps():
    # A backend that internally simulates a long, variable-length rollout still yields
    # exactly ONE grader call per episode — proving grader count tracks episodes, not
    # steps (the 4M-average pathology would scale with steps).
    calls = {"n": 0}
    steps_taken = {"n": 0}

    def spy_grader(*a, **k):
        calls["n"] += 1
        return 1.0

    def stepful_backend(episode_seed, repeat_idx, task):
        for _ in range(100 + episode_seed * 50):  # variable-length "step loop"
            steps_taken["n"] += 1
        return _full_stack_outcome()

    run_contract(_seeds(4), "T4", 1, stepful_backend, grader=spy_grader)
    assert calls["n"] == 4
    assert steps_taken["n"] > 400  # rollouts really did many steps...
    assert calls["n"] == 4  # ...yet grader fired once per episode, not per step


# --- track->task binding owned by the contract layer (SSOT) ------------------


def test_backend_receives_authoritative_task_for_track():
    seen = {}

    def backend(episode_seed, repeat_idx, task):
        seen["task"] = task
        return _full_stack_outcome()

    run_contract(_seeds(1), "T4", 1, backend)
    assert seen["task"] == TASK_BY_TRACK["T4"] == "stack_bowls_three"

    seen.clear()
    run_contract(_seeds(1), "T1", 1, backend)
    assert seen["task"] == TASK_BY_TRACK["T1"] == "adjust_bottle"


# --- nested-mean aggregation over repeats ------------------------------------


def test_repeat_nested_mean():
    # repeat 0: seeds {T,F} -> SR 0.5 ; repeat 1: seeds {T,T} -> SR 1.0 ; mean 0.75
    pattern = {0: [True, False], 1: [True, True]}

    def backend(episode_seed, repeat_idx, task):
        return EpisodeOutcome(success=pattern[repeat_idx][episode_seed])

    res = run_contract(_seeds(2), "T3", 2, backend)
    assert res["per_repeat"] == [0.5, 1.0]
    assert res["sr"] == 0.75
    assert res["n_repeats"] == 2


# --- producer-side fail-loud guards ------------------------------------------


def test_success_must_be_strict_bool():
    # a backend bug handing back a truthy string must NOT count as success
    bad_str = lambda s, r, task: EpisodeOutcome(success="False")  # noqa: E731
    with pytest.raises(TypeError):
        run_contract(_seeds(1), "T1", 1, bad_str)
    # int 1 is not a bool either (would otherwise sneak through `if x:`)
    bad_int = lambda s, r, task: EpisodeOutcome(success=1)  # noqa: E731
    with pytest.raises(TypeError):
        run_contract(_seeds(1), "T1", 1, bad_int)


def test_t4_graded_nan_is_rejected():
    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    with pytest.raises(ValueError):
        run_contract(_seeds(1), "T4", 1, backend, grader=lambda *a, **k: float("nan"))


def test_t4_graded_out_of_range_is_rejected():
    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    with pytest.raises(ValueError):
        run_contract(_seeds(1), "T4", 1, backend, grader=lambda *a, **k: 2.0)


def test_t4_graded_bool_is_rejected():
    # the retired pi05 contract emitted graded as a bool; producer must reject it
    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    with pytest.raises(TypeError):
        run_contract(_seeds(1), "T4", 1, backend, grader=lambda *a, **k: True)


def test_t4_graded_accepts_non_python_real():
    # _check_graded_value accepts any numbers.Real (e.g. a NumPy/Fraction scalar from a
    # custom grader), coercing to a plain float — not just Python int/float.
    from fractions import Fraction

    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    res = run_contract(_seeds(1), "T4", 1, backend, grader=lambda *a, **k: Fraction(1, 2))
    assert res["graded"] == 0.5
    assert isinstance(res["graded"], float)


def test_t4_gripper_flag_must_be_strict_bool():
    # gripper-open flags feed graded_stack_score's `and`; a truthy non-bool must fail
    # loud, not let a full stack silently score 1.0.
    bad = lambda s, r, task: EpisodeOutcome(  # noqa: E731
        success=True,
        final_bowls=[(CX, CY, H1), (CX, CY, H2), (CX, CY, H3)],
        left_gripper_open="True",
        right_gripper_open=True,
    )
    with pytest.raises(TypeError):
        run_contract(_seeds(1), "T4", 1, bad)


def test_repeats_true_bool_rejected():
    # repeats=True (isinstance(True, int)) must be rejected, consistent with episode_seed
    with pytest.raises(ValueError):
        run_contract(_seeds(1), "T1", True, lambda s, r, task: EpisodeOutcome(success=True))


# --- validation --------------------------------------------------------------


def test_unknown_track_rejected():
    with pytest.raises(ValueError):
        run_contract(_seeds(1), "T9", 1, lambda s, r, task: _full_stack_outcome())


def test_non_positive_repeats_rejected():
    with pytest.raises(ValueError):
        run_contract(_seeds(1), "T4", 0, lambda s, r, task: _full_stack_outcome())


def test_empty_seed_table_rejected():
    with pytest.raises(ValueError):
        run_contract([], "T1", 1, lambda s, r, task: EpisodeOutcome(success=True))


def test_duplicate_seed_rejected():
    dupes = [{"episode_seed": 7}, {"episode_seed": 7}]
    with pytest.raises(ValueError):
        run_contract(dupes, "T1", 1, lambda s, r, task: EpisodeOutcome(success=True))


def test_backend_must_return_episode_outcome():
    with pytest.raises(TypeError):
        run_contract(_seeds(1), "T1", 1, lambda s, r, task: (True, None))


def test_t4_malformed_final_state_fails_loud():
    # T4 backend that forgot to capture the 3 bowls -> grader raises (no silent 0)
    bad = lambda s, r, task: EpisodeOutcome(success=True, left_gripper_open=True,  # noqa: E731
                                            right_gripper_open=True)
    with pytest.raises((ValueError, TypeError)):
        run_contract(_seeds(1), "T4", 1, bad)


# --- result.json round-trip (what the worker reads) --------------------------


def test_write_result_json_contains_graded(tmp_path):
    backend = lambda s, r, task: _full_stack_outcome()  # noqa: E731
    res = run_contract(_seeds(2), "T4", 1, backend)
    out = tmp_path / "result.json"
    write_result(res, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["track"] == "T4"
    assert loaded["graded"] == 1.0
    assert set(loaded) >= {"sr", "n_repeats", "n_episodes", "per_repeat", "track", "graded"}
    # standard JSON only — no NaN/Infinity tokens that strict parsers reject
    assert math.isfinite(loaded["graded"])
