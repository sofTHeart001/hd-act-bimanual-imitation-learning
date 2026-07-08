#!/usr/bin/env python3
"""Production ACT in-process eval -> result contract (route ①, replaces pi05 serve_policy).

The evald worker spawns THIS as a subprocess. It runs ACT entirely in-process on one
GPU (no JAX serve_policy, no cross-process bridge — that pi05-era machinery is retired),
drives `act_contract.run_contract` with a real RoboTwin rollout backend, and writes a
`result.json` contract `{sr, n_repeats, n_episodes, per_repeat, track, graded}`. For T4
(stack_bowls_three) it captures the episode-END bowl poses + gripper-open flags +
table_z_bias and `act_contract` calls `graded_stack_score` ONCE per episode. The sim/torch
rollout stays behind `run_contract`'s injection boundary; the leaderboard never imports it.

Reuse strategy: this drives RoboTwin's own ACT model/config machinery via the upstream
`eval_policy` helpers (`class_decorator`, `eval_function_decorator`) and replicates only the
deterministic config assembly (with defensive defaults), so it does NOT depend on the
upstream seed-skipping rollout loop. The per-episode rollout mirrors `eval_policy`'s policy
block (get_obs -> eval -> eval_success latch) MINUS the expert pass (clean configs set
expert_check=false; vanilla ACT has no language conditioning) and MINUS video.

--probe: set up ONE seed and print the T4 final-state fields (bowl poses, gripper vals,
table_z_bias) without building the model or rolling out — grounds the gripper-open threshold
and the env API before trusting graded.

SECURITY — route ① collapses the trust boundary. DECIDED: accept in-process eval + organizer
re-eval; do NOT re-introduce IPC isolation. The mitigations below are the resolved posture, not
open questions:
  * Secret/seed reading: player policy code runs IN THIS PROCESS, so it can read the private seed
    table and the env. Seed privacy rests on (a) the deploy sandbox having NO network egress — a
    HARD precondition, so seeds read in-process cannot be exfiltrated — (b) worker logs/artifacts
    not returned to players, and (c) the worker's eval_uid/eval_gid downgrade (protects
    DB/secrets/other teams). This runner forces policy_name=ACT so a deploy yaml can't widen the
    import surface.
  * Result INTEGRITY: because the same process runs untrusted policy code AND computes/prints the
    result contract, a malicious policy/ckpt could monkeypatch run_contract/write_result/
    eval_success or atexit-forge {sr, graded}. ACCEPTED residual risk; compensating control = the
    main board is graded (must correspond to real physical stacking, hard to fabricate) + the
    organizer runs a controlled re-eval of the top finalists before locking final rankings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from act_contract import (  # noqa: E402
    EpisodeOutcome,
    GRADED_TRACK,
    TASK_BY_TRACK,
    run_contract,
    validate_seed_table,
    write_result,
)


def subrepo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def ensure_local_runtime(robotwin_root: Path) -> Path:
    root = resolve_path(robotwin_root)
    shared = resolve_path(subrepo_root() / "external" / "robotwin")
    if root == shared or shared in root.parents:
        raise SystemExit(f"refusing shared upstream RoboTwin checkout: {root}")
    if not (root / "script" / "eval_policy.py").is_file():
        raise SystemExit(f"RoboTwin eval_policy.py not found under: {root}")
    return root


def load_seed_table(path: str) -> list[dict]:
    with open(path) as f:
        table = json.load(f)
    validate_seed_table(table)
    return table


def get_embodiment_config(robot_file: str) -> dict:
    import yaml

    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_configs(ep, task_name, task_config, ckpt_setting, ckpt_dir, state_dim, temporal_agg,
                  deploy_config="policy/ACT/deploy_policy.yml"):
    """Assemble (usr_args = policy/deploy config, args = task/env config) the way
    eval_policy.main does, but with defensive defaults and full control of eval-only knobs.
    Reuses upstream helpers via the imported `ep` (eval_policy) module + envs.CONFIGS_PATH.

    deploy_config picks the ACT architecture (hidden_dim / chunk_size / camera_names). It MUST
    match the architecture the ckpt was trained with — e.g. the 768-dim / chunk-100 'big'
    models need policy/ACT/deploy_big.yml, not the 512/50 default. A mismatch surfaces as a
    state_dict size-mismatch at load (caught loud, not silently wrong)."""
    import yaml
    from envs import CONFIGS_PATH

    with open(deploy_config, "r", encoding="utf-8") as f:
        usr = yaml.load(f.read(), Loader=yaml.FullLoader)
    usr.update(
        dict(
            task_name=task_name,
            task_config=task_config,
            ckpt_setting=ckpt_setting,
            ckpt_dir=ckpt_dir,
            state_dim=state_dim,
            temporal_agg=temporal_agg,
            seed=0,
        )
    )
    # FORCE policy_name=ACT (codex H2): the deploy yaml must NOT be able to redirect the
    # eval_function_decorator import to an arbitrary module — route ① only ever runs the ACT
    # policy (a player's code package shadows policy/ACT by name via PYTHONPATH, still "ACT").
    usr["policy_name"] = "ACT"
    usr.setdefault("instruction_type", "unseen")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    # The backend deliberately skips the upstream expert pass + instruction (vanilla ACT, no
    # language). eval_policy defaults expert_check=True, so fail LOUD if a config still expects
    # the expert path rather than silently diverging from official eval semantics (codex M4).
    if args.get("expert_check") is not False:
        raise SystemExit(
            f"{task_config}.yml: expert_check must be false for ACT route-① eval, "
            f"got {args.get('expert_check')!r}")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_name"] = usr["policy_name"]

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        cam_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_cam = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam_cfg[head_cam]["h"]
    args["head_camera_w"] = cam_cfg[head_cam]["w"]

    def emb_file(t):
        f = emb_types[t]["file_path"]
        if f is None:
            raise SystemExit(f"no embodiment file for {t!r}")
        return f

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    usr["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # eval-only knobs under our control (never video; render off; required-by-loop defaults).
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None
    args.setdefault("render_freq", 0)
    args["render_freq"] = 0
    args.setdefault("clear_cache_freq", 1)
    return usr, args


def extract_final_state(TASK_ENV, gripper_open_threshold: float) -> dict:
    """T4 episode-END final state -> graded_stack_score kwargs. Read BEFORE close_env.

    gripper-open is derived from the normalized gripper value (get_{left,right}_gripper_val)
    against a threshold; the value range/convention is verified empirically via --probe and
    pinned by the threshold knob. table_z_bias is read straight off the env (0.0 under the
    easy/clean config, which sets random_table_height=0)."""
    bowls = [
        [float(c) for c in TASK_ENV.bowl1.get_pose().p],
        [float(c) for c in TASK_ENV.bowl2.get_pose().p],
        [float(c) for c in TASK_ENV.bowl3.get_pose().p],
    ]
    lg = float(TASK_ENV.robot.get_left_gripper_val())
    rg = float(TASK_ENV.robot.get_right_gripper_val())
    return dict(
        final_bowls=bowls,
        left_gripper_open=bool(lg > gripper_open_threshold),
        right_gripper_open=bool(rg > gripper_open_threshold),
        table_z_bias=float(getattr(TASK_ENV, "table_z_bias", 0.0)),
    )


def make_backend(ep, TASK_ENV, args, model, eval_func, reset_func, track, gripper_open_threshold):
    from envs.utils.create_actor import UnStableError

    graded_board = track == GRADED_TRACK
    off_anchor = [[9.0, 9.0, 0.0], [9.0, 9.0, 0.0], [9.0, 9.0, 0.0]]  # scores graded 0

    def backend(episode_seed: int, repeat_idx: int, task: str) -> EpisodeOutcome:
        # Guard track<->env binding (codex M3): run_contract passes the authoritative
        # TASK_BY_TRACK[track]; refuse to score a different env under this track's contract.
        if task != args["task_name"]:
            raise SystemExit(f"track task {task!r} != configured env task {args['task_name']!r}")
        # mirrors eval_policy's policy block: setup the SAME scene, reset, roll out until the
        # eval_success latch or step_lim. No expert pass / no instruction (clean: expert_check
        # off, vanilla ACT). No seed-skipping (the seed table is pre-validated upstream).
        try:
            TASK_ENV.setup_demo(now_ep_num=0, seed=episode_seed, is_test=True, **args)
        except UnStableError:
            # scene won't settle at this seed -> no stack possible -> failed, graded 0.
            TASK_ENV.close_env()
            if graded_board:
                return EpisodeOutcome(
                    success=False, final_bowls=off_anchor,
                    left_gripper_open=True, right_gripper_open=True, table_z_bias=0.0)
            return EpisodeOutcome(success=False)

        reset_func(model)
        succ = False
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        final = extract_final_state(TASK_ENV, gripper_open_threshold) if graded_board else None
        TASK_ENV.close_env(clear_cache=True)
        if final is not None:
            return EpisodeOutcome(success=bool(succ), **final)
        return EpisodeOutcome(success=bool(succ))

    return backend


def shadow_player_act(player_code_root: str | None) -> None:
    """Make a player's code-package `policy/ACT` shadow the official ACT (codex route① #3).

    The spec requires player code to actually run (§6: "4 题都用选手代码跑"). `import ACT`
    resolves via sys.path, and the trusted root's `policy` dir was put on the path first, so the
    OFFICIAL ACT would load and the player's would be silently ignored. We fix that by putting
    `<player_code_root>/policy` at sys.path[0]. Call this ONLY after the trusted envs/eval_policy
    are already imported (cached in sys.modules), so this can only shadow the not-yet-imported
    top-level `ACT` package — never the trusted sim/env modules.

    fail-LOUD if a code package is given but carries no `policy/ACT/__init__.py`: silently
    falling back to the official ACT (the original bug) must not happen.
    """
    if not player_code_root:
        return
    root = resolve_path(Path(player_code_root))
    act_init = root / "policy" / "ACT" / "__init__.py"
    if not act_init.is_file():
        raise SystemExit(
            f"--player-code-root given but no policy/ACT/__init__.py under {root} — "
            f"player code package must ship a policy/ACT package (it is what runs).")
    sys.path.insert(0, str(root / "policy"))   # player policy dir first -> import ACT = player's
    print(f"[run-act-eval] player ACT shadowing official: {root / 'policy' / 'ACT'}", flush=True)


def _setup_runtime(robotwin_root: str):
    root = ensure_local_runtime(Path(robotwin_root))
    repo = subrepo_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "script"))
    sys.path.insert(0, str(root / "description" / "utils"))
    os.chdir(root)

    from recipes.rollout.tron2_runtime_patch import apply_tron2_runtime_patches

    apply_tron2_runtime_patches()
    # Self-collision patch active -> the setup_demo arm-settle (250 raw steps) is safe in eval.
    os.environ["TRON2_EVAL_SETTLE"] = "1"
    import eval_policy as ep  # noqa: E402  (after path+chdir)

    print("[run-act-eval] Tron2 runtime patches applied (self-collision); eval-settle on",
          flush=True)
    return root, ep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robotwin-root",
                   default=os.environ.get("TRON2_ROBOTWIN_DIR",
                                          str(subrepo_root() / "external" / "robotwin_local")))
    p.add_argument("--track", choices=sorted(TASK_BY_TRACK))
    p.add_argument("--seeds", help="JSON seed table (list of {'episode_seed': int})")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--task-name", default=None, help="default: TASK_BY_TRACK[track]")
    p.add_argument("--task-config", default=None, help="default: <task>_clean")
    p.add_argument("--ckpt-dir", default=None, help="ACT ckpt dir (policy_last.ckpt + dataset_stats.pkl)")
    p.add_argument("--ckpt-setting", default="eval")
    p.add_argument("--deploy-config", default="policy/ACT/deploy_policy.yml",
                   help="ACT arch config (must match ckpt: 'big' 768/100 -> policy/ACT/deploy_big.yml)")
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--temporal-agg", action="store_true")
    p.add_argument("--gripper-open-threshold", type=float, default=0.5)
    p.add_argument("--player-code-root", default=None,
                   help="player code package root; its policy/ACT shadows the official ACT "
                        "(codex route① #3). Omit to run the official ACT.")
    p.add_argument("--out", default="result.json")
    # probe mode
    p.add_argument("--probe", action="store_true",
                   help="set up ONE seed and print T4 final-state fields, then exit")
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--probe-task", default="stack_bowls_three")
    p.add_argument("--probe-task-config", default="stack_bowls_three_clean")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    root, ep = _setup_runtime(a.robotwin_root)

    if a.probe:
        usr, args = build_configs(ep, a.probe_task, a.probe_task_config, "probe",
                                  a.ckpt_dir or "", a.state_dim, a.temporal_agg, a.deploy_config)
        TASK_ENV = ep.class_decorator(args["task_name"])
        TASK_ENV.setup_demo(now_ep_num=0, seed=a.probe_seed, is_test=True, **args)
        state = extract_final_state(TASK_ENV, a.gripper_open_threshold)
        lg = float(TASK_ENV.robot.get_left_gripper_val())
        rg = float(TASK_ENV.robot.get_right_gripper_val())
        print("[probe] seed=%d step_lim=%s" % (a.probe_seed, getattr(TASK_ENV, "step_lim", "?")))
        print("[probe] left_gripper_val=%.4f right_gripper_val=%.4f (threshold=%.2f)"
              % (lg, rg, a.gripper_open_threshold))
        print("[probe] table_z_bias=%r" % state["table_z_bias"])
        print("[probe] bowl_poses(xyz)=" + json.dumps(state["final_bowls"]))
        print("[probe] derived left_open=%s right_open=%s"
              % (state["left_gripper_open"], state["right_gripper_open"]))
        TASK_ENV.close_env()
        return 0

    if not a.track or not a.seeds:
        raise SystemExit("eval mode requires --track and --seeds")
    # Authoritative task is track-derived (codex M3): --task-name is NOT honored in eval, so a
    # T1 contract can never be made to score another env. (--task-name stays a probe-only knob.)
    task_name = TASK_BY_TRACK[a.track]
    task_config = a.task_config or f"{task_name}_clean"
    if a.ckpt_dir is None:
        raise SystemExit("--ckpt-dir is required for eval")

    seed_table = load_seed_table(a.seeds)
    usr, args = build_configs(ep, task_name, task_config, a.ckpt_setting, a.ckpt_dir,
                              a.state_dim, a.temporal_agg, a.deploy_config)

    # Shadow the official ACT with the player's code package (codex route① #3). MUST happen
    # AFTER build_configs (which imports the trusted envs/eval_policy into sys.modules) and
    # BEFORE eval_function_decorator imports ACT — so only the not-yet-imported top-level `ACT`
    # resolves to the player's; trusted envs/sim/eval_policy stay the cached official ones.
    shadow_player_act(a.player_code_root)

    get_model = ep.eval_function_decorator(usr["policy_name"], "get_model")
    eval_func = ep.eval_function_decorator(usr["policy_name"], "eval")
    reset_func = ep.eval_function_decorator(usr["policy_name"], "reset_model")
    model = get_model(usr)
    TASK_ENV = ep.class_decorator(args["task_name"])

    backend = make_backend(ep, TASK_ENV, args, model, eval_func, reset_func,
                           a.track, a.gripper_open_threshold)
    result = run_contract(seed_table, a.track, a.repeats, backend)
    # result契约的权威投递通道 = STDOUT(worker 从 stdout 取分,见 worker HIGH#1)。--out 仅归档:
    # best-effort(codex route① my#3:降权子进程可能写不动 worker-owned 归档目录,绝不能因此
    # 让一次成功 eval 失败 / 在打印 stdout 前崩)。
    try:
        write_result(result, a.out)
    except OSError as e:
        print("[run-act-eval] WARN archival write to %s failed (non-fatal): %r" % (a.out, e),
              file=sys.stderr, flush=True)
    print("[run-act-eval] wrote %s: %s" % (a.out, json.dumps(result)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
