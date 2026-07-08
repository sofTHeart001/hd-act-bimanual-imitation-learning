#!/usr/bin/env python3
"""单条推理 rollout 核查:在指定 seed 上跑一次策略,打印该 episode 末态成绩(sr/graded)。

跑的是与官方评测同一个内核 run_act_eval(单 seed),用于核查行为是否合理。
注:评测内核默认**不渲染视频**(eval_video_log 关),本脚本给的是**数值结果**;
需要画面回放见参赛文档 §02「采集可视化贴士」(render_freq;渲染较重,单独流程)。

用法:
    python starter/watch_rollout.py --track T1 --ckpt-dir ./ckpt/T1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = KIT_ROOT / "recipes" / "eval" / "run_act_eval.py"
ROBOTWIN_ROOT = KIT_ROOT / "external" / "robotwin_local"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=["T1", "T2", "T3", "T4"])
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="单个 episode seed")
    ap.add_argument("--temporal-agg", action="store_true")
    ap.add_argument("--deploy-config", default="policy/ACT/deploy_policy.yml",
                    help="ACT 推理配置(相对 robotwin-root;训 big config 的要对应改)")
    ap.add_argument("--out", default="rollout_result.json")
    a = ap.parse_args()

    if not Path(a.ckpt_dir).exists():
        raise SystemExit(f"[watch_rollout] --ckpt-dir 不存在: {a.ckpt_dir}")

    # 临时单 seed 表(只跑这一条 episode)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([{"episode_seed": a.seed}], fh)
        seed_file = fh.name

    cmd = [
        sys.executable, str(RUNNER),
        "--robotwin-root", str(ROBOTWIN_ROOT),
        "--track", a.track,
        "--seeds", seed_file,
        "--ckpt-dir", str(Path(a.ckpt_dir).resolve()),
        "--deploy-config", a.deploy_config,
        "--repeats", "1",
        "--out", a.out,
    ]
    if a.temporal_agg:
        cmd.append("--temporal-agg")
    print(f"[watch_rollout] track={a.track} seed={a.seed}\n  " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(KIT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
