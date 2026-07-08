"""ACT in-process eval -> result contract producer (GPU-free orchestration core).

The **result contract** `{sr, n_repeats, n_episodes, per_repeat, track, graded}` is
the SINGLE architectural seam of the eval kernel: the GPU/sim side produces it, the
leaderboard backend (`troncamp-organizer`) only ever consumes it and never imports
sim/GPU. This module is the producer's *pure orchestration core* — it loops
`seed_table x repeats` over an **injected rollout backend** and, for the T4 main
board (stack_bowls_three), calls `graded_stack_score` ONCE per episode on the
rollout's final state.

The real backend runs ACT **in-process on a single GPU** (no JAX `serve_policy`, no
cross-process bridge — that pi05-era machinery is retired for ACT). A fake backend
exercises every line here without a GPU, which is why the contract assembly lives in
this module and the sim/torch rollout lives behind the `rollout_backend` boundary.

Contract field semantics (aligns with `troncamp-organizer/leaderboard/scoring_io.py`):
  - `sr`        : binary success-rate mean (all tracks; episode-mean then repeat-mean)
  - `graded`    : **float** in [0,1], the mean per-episode graded /3 score — ONLY for
                  T4. scoring_io ranks the T4 main board by `details.graded`. (NOTE: in
                  the retired pi05 eval-kit, `graded` was a *bool flag*; here it is the
                  float mean. T1-T3 omit the key entirely -> pure SR threshold.)
  - per_repeat  : per-repeat binary SR list (nested-mean noise reduction over rollouts).

CRITICAL (the known 4M-average bug): grading is structurally THE ORCHESTRATOR's job.
The backend returns raw final STATE (bowl poses + gripper flags), never a score, so it
cannot grade per-step; `graded_stack_score` is invoked exactly once per episode from
`run_contract`, never wired into the env's per-step `check_success`. The contract-layer
test here pins "grader calls == episode count"; the stronger env-level guard (that
`graded_stack_score` is unreachable from `check_success`) is a Phase-2 test against the
real ACT backend, not assertable in this GPU-free unit.
"""
from __future__ import annotations

import json
import math
import numbers
import os
import sys
from typing import Callable, NamedTuple, Optional, Sequence

sys.path.insert(0, os.path.dirname(__file__))
from graded_score import graded_stack_score  # noqa: E402

# Track -> RoboTwin task (single source of truth for the ACT 4-task suite).
TASK_BY_TRACK = {
    "T1": "adjust_bottle",
    "T2": "grab_roller",
    "T3": "stack_bowls_two",
    "T4": "stack_bowls_three",
}
# The one graded main-board track. Only this track produces a float `graded`.
GRADED_TRACK = "T4"


class EpisodeOutcome(NamedTuple):
    """One ACT rollout's result, as the injected backend reports it.

    success: binary success latch (`TASK_ENV.eval_success`), all tracks.
    final_bowls / left_gripper_open / right_gripper_open / table_z_bias: T4-only
        final-state inputs to `graded_stack_score`, captured at episode end. They
        stay None for T1-T3 (no grading) and MUST be populated for T4 (else the
        grader fails loud on a malformed final state — by design).
    """

    success: bool
    final_bowls: Optional[Sequence[Sequence[float]]] = None
    left_gripper_open: Optional[bool] = None
    right_gripper_open: Optional[bool] = None
    table_z_bias: float = 0.0


# A rollout backend is any callable: (episode_seed, repeat_idx, task) -> EpisodeOutcome.
# `task` is the authoritative RoboTwin task for this track (TASK_BY_TRACK[track]),
# threaded through by the producer so the contract layer — not the backend — owns the
# track->task binding. The backend uses/asserts it (a backend wired to the wrong env
# can fail loud instead of silently scoring track="T4" on the wrong task).
RolloutBackend = Callable[[int, int, str], EpisodeOutcome]


def _require_bool(name: str, value) -> bool:
    """Strict Python-bool guard for binary final-state flags. A backend bug handing back
    a truthy string/int must fail loud, not silently flip a gate or a graded layer (e.g.
    gripper_open='False' would otherwise let a full stack score 1.0)."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {value!r}")
    return value


def _check_graded_value(g):
    """Fail-LOUD producer-side guard: a per-episode graded score must be a non-bool
    finite real in [0,1]. Defense in depth — scoring_io.coerce_graded fail-CLOSES on
    bad values downstream, but the producer must never emit a bad contract (a NaN would
    serialize to non-standard JSON `NaN`; a 2.0 would be silently dropped at the board).
    Accepts any real (incl. NumPy scalars) but rejects bool."""
    if isinstance(g, bool) or not isinstance(g, numbers.Real):
        raise TypeError(f"graded must be a real number, got {g!r}")
    g = float(g)
    if not math.isfinite(g) or not (0.0 <= g <= 1.0):
        raise ValueError(f"graded must be finite in [0,1], got {g!r}")
    return g


def validate_seed_table(seed_table) -> None:
    """A seed table is a non-empty list of {'episode_seed': int} with unique seeds."""
    if not isinstance(seed_table, list) or not seed_table:
        raise ValueError("seed table must be a non-empty list")
    seen = set()
    for i, entry in enumerate(seed_table):
        if not isinstance(entry, dict) or "episode_seed" not in entry:
            raise ValueError(f"seed table row {i} must be a dict with 'episode_seed'")
        s = entry["episode_seed"]
        if not isinstance(s, int) or isinstance(s, bool):
            raise ValueError(f"seed table row {i}: episode_seed must be int, got {s!r}")
        if s in seen:
            raise ValueError(f"seed table has duplicate episode_seed {s}")
        seen.add(s)


def _means_over_repeats(per_repeat_scores):
    """Episode-mean each repeat, returning (per_repeat_means, n_episodes).

    Requires every repeat to cover the same (rectangular) episode count, so a
    ragged backend can't be silently averaged into a plausible-looking number.
    """
    if not per_repeat_scores:
        raise ValueError("no repeats")
    lengths = {len(ep) for ep in per_repeat_scores}
    if len(lengths) != 1:
        raise ValueError(f"ragged repeats, episode counts = {sorted(lengths)}")
    (n_episodes,) = lengths
    if n_episodes == 0:
        raise ValueError("repeats have zero episodes")
    per_repeat = [sum(ep) / len(ep) for ep in per_repeat_scores]
    return per_repeat, n_episodes


def run_contract(
    seed_table,
    track: str,
    repeats: int,
    rollout_backend: RolloutBackend,
    *,
    grader: Callable[..., float] = graded_stack_score,
) -> dict:
    """Drive the injected backend over seed_table x repeats -> result contract dict.

    grader is injectable purely so tests can spy on the call count; production uses
    the real `graded_stack_score`. For T4 the grader is invoked once per episode on
    the rollout's final state; for T1-T3 it is never invoked.
    """
    if track not in TASK_BY_TRACK:
        raise ValueError(f"unknown track {track!r}, expected one of {sorted(TASK_BY_TRACK)}")
    validate_seed_table(seed_table)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError(f"repeats must be a positive int, got {repeats!r}")

    task = TASK_BY_TRACK[track]
    graded_board = track == GRADED_TRACK
    per_repeat_success: list[list[float]] = []
    per_repeat_graded: list[list[float]] = []

    for r in range(repeats):
        succ_scores: list[float] = []
        graded_scores: list[float] = []
        for entry in seed_table:
            outcome = rollout_backend(entry["episode_seed"], r, task)
            if not isinstance(outcome, EpisodeOutcome):
                raise TypeError(
                    f"rollout_backend must return EpisodeOutcome, got {type(outcome).__name__}"
                )
            # Strict bool: a backend bug handing back a truthy string/int must NOT
            # silently count as success and pollute a pass/fail gate. The real backend
            # hands a Python bool latch (TASK_ENV.eval_success), same as rollout_one_seed.
            _require_bool("EpisodeOutcome.success", outcome.success)
            succ_scores.append(1.0 if outcome.success else 0.0)
            if graded_board:
                graded_scores.append(
                    _check_graded_value(
                        grader(
                            outcome.final_bowls,
                            table_z_bias=outcome.table_z_bias,
                            left_gripper_open=_require_bool(
                                "left_gripper_open", outcome.left_gripper_open),
                            right_gripper_open=_require_bool(
                                "right_gripper_open", outcome.right_gripper_open),
                        )
                    )
                )
        per_repeat_success.append(succ_scores)
        if graded_board:
            per_repeat_graded.append(graded_scores)

    per_repeat_sr, n_episodes = _means_over_repeats(per_repeat_success)
    result = {
        "sr": sum(per_repeat_sr) / len(per_repeat_sr),
        "n_repeats": len(per_repeat_success),
        "n_episodes": n_episodes,
        "per_repeat": per_repeat_sr,
        "track": track,
    }
    if graded_board:
        per_repeat_g, _ = _means_over_repeats(per_repeat_graded)
        result["graded"] = sum(per_repeat_g) / len(per_repeat_g)
    return result


def write_result(result: dict, path: str) -> None:
    """Persist the result contract as result.json (what the evald worker reads)."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
