#!/usr/bin/env python3
"""选手本地自评:在公开 seed 上跑主办方同一套评测内核(run_act_eval),输出 sr / graded。

- 跑的就是官方评测的同一个内核 `recipes/eval/run_act_eval.py`,**同内核、同任务、同分布,
  只换种子**:官方榜(dev/final)用主办方私有 100 seed、每 seed **单次 rollout(`--repeats 1`,
  与本地默认同口径)**,选手拿不到私有 seed(防过拟合)。本地公开 100 seed 的分数是官方分的
  无偏估计,**不保证逐分相等**(种子集不同 + 单次 rollout 的采样噪声)。
- 输出 result 契约 {sr, n_episodes, n_repeats, per_repeat, track, graded}。T4 看 graded(/100
  口径在榜单换算:graded∈[0,1]×100;三层各 1/3 相加、叠满 100)。

用法:
    python starter/eval_local.py --track T1 --ckpt-dir ./ckpt/T1
    python starter/eval_local.py --track T4 --ckpt-dir ./ckpt/T4 --temporal-agg
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]            # starter/ 的上一层 = 选手包根
RUNNER = KIT_ROOT / "recipes" / "eval" / "run_act_eval.py"  # 官方评测内核(唯一通道)
ROBOTWIN_ROOT = KIT_ROOT / "external" / "robotwin_local"
PUBLIC_SEEDS = KIT_ROOT / "starter" / "public_seeds.json"   # 随包下发的公开 100 seed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=["T1", "T2", "T3", "T4"],
                    help="赛道(任务由 track 决定:T1 adjust_bottle / T2 grab_roller / "
                         "T3 stack_bowls_two / T4 stack_bowls_three)")
    ap.add_argument("--ckpt-dir", required=True,
                    help="ACT ckpt 目录(policy_last.ckpt + dataset_stats.pkl)")
    ap.add_argument("--seeds", default=str(PUBLIC_SEEDS),
                    help="seed 表 JSON(默认随包公开 100 seed;官方评测用主办方私有 seed,"
                         "同分布不同种子)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="同 policy 重复 rollout 取均值消 ACT 采样噪声。默认 1,与官方评测同口径"
                         "(EVAL_REPEATS=1,单次 rollout);调大只用于本地自查稳定性,不再贴合官方单次口径")
    ap.add_argument("--temporal-agg", action="store_true",
                    help="推理时闭环(temporal aggregation;T4 可用)")
    ap.add_argument("--deploy-config", default="policy/ACT/deploy_policy.yml",
                    help="ACT 推理配置(相对 robotwin-root)")
    ap.add_argument("--out", default="result.json", help="result 契约落盘路径")
    a = ap.parse_args()

    if not RUNNER.exists():
        raise SystemExit(f"[eval_local] 找不到评测内核 {RUNNER};选手包是否完整(setup 装好)?")
    if not Path(a.ckpt_dir).exists():
        raise SystemExit(f"[eval_local] --ckpt-dir 不存在: {a.ckpt_dir}")
    if not Path(a.seeds).exists():
        raise SystemExit(f"[eval_local] seed 表不存在: {a.seeds}")

    cmd = [
        sys.executable, str(RUNNER),
        "--robotwin-root", str(ROBOTWIN_ROOT),
        "--track", a.track,
        "--seeds", str(Path(a.seeds).resolve()),
        "--ckpt-dir", str(Path(a.ckpt_dir).resolve()),
        "--deploy-config", a.deploy_config,
        "--repeats", str(a.repeats),
        "--out", a.out,
    ]
    if a.temporal_agg:
        cmd.append("--temporal-agg")
    print("[eval_local] 跑官方评测内核:\n  " + " ".join(cmd), flush=True)
    # 在包根下跑(run_act_eval 自解析同目录 import + 用 --robotwin-root 定位 RoboTwin)。
    return subprocess.call(cmd, cwd=str(KIT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
