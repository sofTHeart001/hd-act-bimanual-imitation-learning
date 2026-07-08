#!/usr/bin/env python3
"""TronCamp 选手环境自检(单一 conda 环境:RoboTwin 仿真 + ACT 训练共用)。

核心目的两条:
  1) 核心仿真 / 训练依赖能 import(区分"没装"和"装坏了 CUDA/.so")。
  2) **断言 curobo == 0.8.0**:选手包内嵌的是 cuRobo 0.8.0(Apache-2.0)纯源码。若探到
     0.7.x(旧版残留在环境里),立刻 fail —— 新版 planner.py 用的是 0.8.0 的 API
     (MotionPlanner / tool_frames / update_tool_pose_criteria),跑在 0.7.x 上会
     import/行为错乱。

用法:
    python setup/env_check.py            # 全量自检
    python setup/env_check.py --quiet    # 只在失败时输出
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

# 单一环境:RoboTwin 仿真 + ACT 训练都在同一个 conda env。
# einops 是 ACT 训练栈(policy/ACT/imitate_episodes.py)的顶层硬 import,RoboTwin 自带
# requirements 不含它 —— 缺它 turnkey `train.sh` 秒崩,故列入自检(B9)。
REQUIRED_MODULES = ["curobo", "sapien", "mplib", "torch", "numpy", "h5py", "einops"]

# curobo 版本要求:必须是 0.8.0 发布线(内嵌 v0.8.0);明确拒绝 0.7.x 残留。
CUROBO_EXPECTED = "0.8.0"
CUROBO_REJECT_PREFIX = "0.7"

# 打包占位符:build 时把机器绝对路径换成 __KIT_ROOT__,装机第 6 步须 sed 还原成本仓实际路径。
# 漏这步 curobo YAML 里仍是占位串,规划器加载 collision / 描述文件会失败。
# 占位只落在 *.yml(由 build 占位契约 gate 保证);故这里全树扫 *.yml,与装机第 6 步
# `--include='*.yml'` 的还原范围、build gate 的校验范围三者同源——新增被占位的 yml 也自动覆盖。
KIT_ROOT = Path(__file__).resolve().parents[1]  # setup/ 的上一层 = 选手包根
# 探测基准拆写:装机第 6 步已限定 `--include='*.yml'`,正常流程不碰本 .py;但若有人误跑旧的
# 「全树 sed」(不带 --include)扫到本文件,这行不会被替换成真实路径——否则常量被改后会把
# 已还原的 yml 反判为「未还原」(即本次修复的原始 bug)。拼接结果等价 "__KIT_ROOT__"。
KIT_ROOT_PLACEHOLDER = "__KIT" + "_ROOT__"


def _import_report(modules: list[str]) -> dict[str, tuple[bool, str]]:
    """逐模块 import,返回 {mod: (ok, detail)}。"""
    out: dict[str, tuple[bool, str]] = {}
    for m in modules:
        try:
            importlib.import_module(m)
            out[m] = (True, "")
        except ModuleNotFoundError as e:
            out[m] = (False, f"未安装 ({e})")
        except Exception as e:  # noqa: BLE001 — 装坏(CUDA/.so 等)也要报出来
            out[m] = (False, f"装坏: {type(e).__name__}: {e}")
    return out


def _curobo_dist_version() -> str | None:
    """取**已安装 dist 的元数据版本**(而非 curobo.__version__ —— 后者在无 git 源码树里会
    回落到硬编码 'v0.8.0-no-tag',探不出真正装了什么)。取不到返回 None。"""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        return None
    for name in ("nvidia-curobo", "nvidia_curobo"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def check_curobo_version() -> tuple[bool, str]:
    """断言已安装的 curobo 是 0.8.0 发布线。返回 (ok, detail)。以 dist 元数据为准。"""
    raw = _curobo_dist_version()
    if raw is None:
        return False, (
            "找不到 nvidia-curobo 的安装元数据 —— 疑似未正确 pip install envs/curobo。"
            "请见安装文档分步重装 curobo(须带 SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0)。"
        )
    ver = str(raw).lstrip("vV").strip()
    if ver.startswith(CUROBO_REJECT_PREFIX):
        return False, (
            f"探到 curobo {ver} —— 旧版残留。选手包要求 {CUROBO_EXPECTED}(Apache-2.0)。"
            "请卸载旧版后见安装文档分步重装内嵌的 envs/curobo。"
        )
    if "no-tag" in ver:
        return False, (
            f"curobo 版本探测异常({ver}),疑似安装元数据缺失。"
            "请见安装文档分步重装 curobo(带 SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0)。"
        )
    # 接受 0.8.0 及其 dev/post/local 变体(pretend 版本固定 0.8.0),拒绝其他(含 0.8.1)。
    if not (ver == CUROBO_EXPECTED or ver.startswith(CUROBO_EXPECTED + ".")
            or ver.startswith(CUROBO_EXPECTED + "+") or ver.startswith(CUROBO_EXPECTED + "post")):
        return False, f"探到 curobo {ver},要求 {CUROBO_EXPECTED}(内嵌 v0.8.0)"
    return True, ver


def check_ffmpeg() -> tuple[bool, str]:
    """采集保存链会 subprocess 调 `ffmpeg` 二进制合成 episode 视频;缺它成功集会静默保存失败、
    采集计数卡 0。这里断言 `ffmpeg -version` 可执行。返回 (ok, detail)。"""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False, (
            "PATH 里找不到 ffmpeg。采集保存视频需要它。见安装文档分步重做 ffmpeg 兜底"
            "(用 imageio-ffmpeg 静态二进制),或手动装 ffmpeg 到当前 conda 环境。"
        )
    try:
        subprocess.run([exe, "-version"], capture_output=True, check=True)
    except Exception as e:  # noqa: BLE001 — 二进制在但跑不起来(缺库等)也要报
        return False, f"ffmpeg 存在({exe})但 `ffmpeg -version` 执行失败:{e}"
    return True, exe


def check_kit_root_placeholder() -> tuple[bool, str]:
    """全树扫描 *.yml,断言无 __KIT_ROOT__ 占位残留(装机第 6 步须还原)。与第 6 步
    `--include='*.yml'` 的还原范围、build 占位契约 gate 的校验范围同源。命中即 fail。返回 (ok, detail)。"""
    hit: list[str] = []
    scanned = 0
    for fp in KIT_ROOT.rglob("*.yml"):
        if ".git" in fp.parts:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        if KIT_ROOT_PLACEHOLDER in text:
            hit.append(str(fp.relative_to(KIT_ROOT)))
    if hit:
        shown = "、".join(hit[:8]) + ("…" if len(hit) > 8 else "")
        return False, (
            "未完成安装第 6 步 __KIT_ROOT__ 占位还原,curobo 规划器会起不来 —— 以下 yml 仍含占位串:"
            + shown
            + '。请在选手包根目录跑:grep -rl --include="*.yml" __KIT_ROOT__ . | xargs -r sed -i "s#__KIT_ROOT__#$(pwd)#g"'
        )
    if scanned == 0:
        return True, "未找到 yml(跳过)"
    return True, f"{scanned} 个 yml 无残留占位"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TronCamp 选手环境自检(单一 conda 环境:RoboTwin 仿真 + ACT 训练共用)")
    ap.add_argument("--quiet", action="store_true", help="只在失败时输出")
    a = ap.parse_args(argv)

    problems: list[str] = []

    # 1) 依赖 import
    report = _import_report(REQUIRED_MODULES)
    for m, (ok, detail) in report.items():
        if not ok:
            problems.append(f"{m}: {detail}")
        if not a.quiet:
            print(f"[{'OK' if ok else 'FAIL'}] import {m}" + (f"  — {detail}" if detail else ""))

    # 2) curobo 版本断言(仅当 curobo 本身 import 成功时才有意义)
    if report.get("curobo", (False, ""))[0]:
        ok, detail = check_curobo_version()
        if not ok:
            problems.append(f"curobo 版本: {detail}")
        if not a.quiet:
            print(f"[{'OK' if ok else 'FAIL'}] curobo 版本" + (f"  — {detail}" if detail else ""))

    # 3) ffmpeg 可执行(采集视频保存链依赖)
    ok, detail = check_ffmpeg()
    if not ok:
        problems.append(f"ffmpeg: {detail}")
    if not a.quiet:
        print(f"[{'OK' if ok else 'FAIL'}] ffmpeg" + (f"  — {detail}" if detail else ""))

    # 4) __KIT_ROOT__ 占位还原(安装第 6 步)—— 全树扫 *.yml,断言无残留占位
    ok, detail = check_kit_root_placeholder()
    if not ok:
        problems.append(f"__KIT_ROOT__ 占位: {detail}")
    if not a.quiet:
        print(f"[{'OK' if ok else 'FAIL'}] __KIT_ROOT__ 占位还原" + (f"  — {detail}" if detail else ""))

    if problems:
        print("\n环境自检未通过:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if not a.quiet:
        print("\n环境全部就绪(依赖 import 通过,curobo 0.8.0 已就位,__KIT_ROOT__ 占位已还原)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
