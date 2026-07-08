"""T4 (stack_bowls_three) graded score — pure function, episode end ONLY.

Computes a /3 stacking-progress score for the main-board task. 3/3 is exactly
binary success tightened to the anchor C (the marked stack center). Partial
credit (1/3, 2/3) gives the leaderboard the resolution that binary success rate
lacks at ~20% Tron2 SR.

CRITICAL: call this ONCE on the final state, from the eval runner. Do NOT wire
it into `check_success` (which the rollout loop calls every step) — averaging a
per-step graded value over ~millions of calls yields a meaningless number. See
spec §3 (docs/superpowers/specs/2026-06-29-troncamp-act-4task-suite-design.md).

Pure Python (no numpy/sim) so it unit-tests anywhere.
"""

ANCHOR_C = (0.0, -0.1)          # target stack center (= expert bowl1_target_pose xy)
LAYER_HEIGHTS = (0.74, 0.77, 0.81)  # base / middle / top, before table_z_bias
EPS_XY = 0.04
EPS_Z = 0.02
# Min vertical separation between consecutively-matched layers. The height bands
# overlap (EPS_Z=0.02 vs 0.03 layer spacing), so without this two bowls at the
# same z near the band overlap could both be credited (base + middle). Real
# nested bowls sit ~0.03 apart, far above this floor.
MIN_LAYER_GAP = 0.01


def graded_stack_score(
    bowl_positions,
    *,
    table_z_bias=0.0,
    left_gripper_open,
    right_gripper_open,
):
    """Return the graded stacking score in {0, 1/3, 2/3, 1}.

    bowl_positions: iterable of exactly three (x, y, z) world positions, taken
        at episode end. Each item must have at least 3 components.
    table_z_bias: added to every layer height (the env's table-height offset).
    left/right_gripper_open: final gripper states (REQUIRED, keyword-only). A
        held bowl must never be scored as placed, so the caller must pass these
        explicitly rather than relying on a default. Layers 2 and 3 only count
        when BOTH grippers are open — a bowl held in a closed gripper at layer
        height is not "placed" (review: holding at layer-2 height used to earn
        2/3). Layer 1 is exempt: it sits at table height, where holding is no
        better than the already-accepted shoved-to-C case (1/3 cap below).

    Only bowls within EPS_XY of the anchor C (both axes) can be part of the
    stack; those are sorted by height and matched bottom-up to the layer heights
    (lowest -> base). Sorting — rather than first-match — makes the score
    independent of input order even where adjacent layer height bands overlap
    (0.74±0.02 and 0.77±0.02 share [0.75, 0.76]). The score is the number of
    consecutively satisfied layers / 3: stopping at the first unsatisfied layer
    caps gaming (a bowl shoved to C on the table earns at most 1/3) and enforces
    the anchor (a perfect stack offset from C earns 0).

    NOTE: 3/3 is binary success *tightened* to absolute anchor C with a two-sided
    height tolerance — intentionally STRICTER than the env's legacy check_success
    (relative xy + one-sided `z - target < eps`), not identical to it.
    """
    bowls = [tuple(p) for p in bowl_positions]
    if len(bowls) != 3:
        raise ValueError("expected exactly 3 bowl positions")
    if any(len(p) < 3 for p in bowls):
        raise ValueError("each bowl position needs (x, y, z)")
    heights = [h + table_z_bias for h in LAYER_HEIGHTS]

    def at_anchor(p):
        return abs(p[0] - ANCHOR_C[0]) < EPS_XY and abs(p[1] - ANCHOR_C[1]) < EPS_XY

    anchor_bowls = sorted((p for p in bowls if at_anchor(p)), key=lambda p: p[2])

    layers = 0
    for i, h in enumerate(heights):
        if i >= len(anchor_bowls) or abs(anchor_bowls[i][2] - h) >= EPS_Z:
            break
        if i >= 1 and anchor_bowls[i][2] - anchor_bowls[i - 1][2] < MIN_LAYER_GAP:
            break  # no real vertical separation -> not a distinct higher layer
        if i >= 1 and not (left_gripper_open and right_gripper_open):
            break  # held-at-height must not score: layers 2+ require open grippers
        layers = i + 1
    return layers / 3.0
