#!/usr/bin/env python3
"""选手提交 CLI(唯一提交通道)。

把 ckpt(+ T2/T3/T4 的整库代码包)+ token 上传后端,排进评测队列。流程:
  1) 上传 ckpt 文件到 /api/upload → 拿 file_url + 服务端 sha256;
  2)(T2/T3/T4)打包 --code-dir 整库为 .tar.gz,上传 → 拿 code file_url;
  3) /api/submit 带 token + track + ckpt_url + ckpt_sha256 (+ code_url)。

T1 仅交权重(自采数据 + 官方 config 的入门保底题);T2/T3/T4 交权重 + 代码包。
顺序解锁 T1→T2→T3→T4(后端 gate)。token 走 header/body,不落榜(榜上只有尾号)。

注:本 CLI 不依赖评测内核;ckpt 只打 {你的 .ckpt→policy_last.ckpt, dataset_stats.pkl} 两文件,
代码包打 --code-dir 整库并做基本排除(.git/__pycache__/产物目录)。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

# 官方 mani 提交入口(HTTPS 反代;token 全程 TLS)。优先级:--server > env TRONCAMP_SERVER > 此默认。
DEFAULT_SERVER = "https://submit.troncamp-mani.limxdynamics.com"

# junk + 训练产物名:任意层级都排。act_ckpt/ckpt/checkpoints/datasets/processed_data/
# eval_data/eval_result/results/runs/uploads 都是约定俗成的「产物/输出」目录名,嵌套出现
# (如 policy/pi05/checkpoints)也是冗余 → 任意层排,堵 codex 复审指出的嵌套 bloat 洞;
# ckpt 本已单独上传,打进代码包纯冗余。
_EXCLUDE_ANYWHERE = {".git", "__pycache__", ".venv", ".venv-test", "node_modules",
                     "act_ckpt", "ckpt", "checkpoints", "datasets",
                     "processed_data", "eval_data", "eval_result", "results", "runs", "uploads"}
# 仅在代码包根(顶层)排,避免误删选手嵌套的同名源码模块:
# - data:通用名,选手可能嵌套真源码(如 T4 改造新增 policy/ACT/data/)→ 只顶层排。
# - assets/envs/description:RoboTwin 根下的官方重型平台目录(assets~188M、envs/curobo~176M)。
#   worker 评测只把 <repo_root>/policy 插到 sys.path[0] 遮蔽官方 ACT,这些目录评测一律用官方
#   eval root 的版本、选手副本 worker 从不读 → 顶层排除把代码包从 ~376M 压到 ~8.5M(policy/);
#   只顶层排(不排嵌套),保留选手可能放在 policy/ACT/ 下的同名模型资产。
_EXCLUDE_TOPLEVEL = {"data", "assets", "envs", "description"}
# 向后兼容:旧引用 _EXCLUDE_DIRS 取并集
_EXCLUDE_DIRS = _EXCLUDE_ANYWHERE | _EXCLUDE_TOPLEVEL


# 后端 HTTP 错误 → 友好中文提示(选手看不懂原始 urllib.error.HTTPError traceback)。
_HTTP_HINTS = {
    400: "请求不合法(400):字段/参数有误,请更新选手包或检查 --track / --ckpt / --code-dir",
    401: "令牌无效(401):检查 --token / 环境变量 TRONCAMP_TOKEN / --token-file",
    403: "赛道未解锁(403):需按 T1→T2→T3→T4 顺序解锁,请先通过上一题(如交 T2 前须先过 T1)",
    404: "地址不对(404):检查 --server 是否为主办方公布的提交地址",
    409: "赛道已锁定(409):该赛道已达标锁定,不能重复提交",
    413: "文件过大(413):ckpt / 代码包超出后端上限",
    429: "过于频繁(429):提交有频率限制,请稍后再试",
}


def _raise_http_error(url: str, e: urllib.error.HTTPError):
    """把后端 HTTP 错误转成友好中文提示并非零退出(读 body 附上后端明细)。"""
    detail = ""
    try:
        raw = e.read().decode("utf-8", "replace").strip()
    except Exception:
        raw = ""
    if raw:
        try:
            j = json.loads(raw)
            detail = str(j.get("error") or j.get("message") or j.get("detail") or raw)
        except Exception:
            detail = raw
    msg = _HTTP_HINTS.get(e.code, f"提交失败(HTTP {e.code} {e.reason})")
    tail = f"\n  后端信息:{detail}" if detail else ""
    raise SystemExit(f"[submit] ✗ {msg}\n  ({url}){tail}")


def _post_multipart(url: str, fields: dict, file_field: str | None,
                    file_path: Path | None, file_bytes: bytes | None,
                    file_name: str | None) -> dict:
    """极简 multipart/form-data POST(只用标准库,选手机器零额外依赖)。"""
    boundary = "----troncampsubmit0xBEEF"
    body = io.BytesIO()

    def w(s: str):
        body.write(s.encode("utf-8"))

    for k, v in fields.items():
        w(f"--{boundary}\r\n")
        w(f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    if file_field is not None:
        data = file_bytes if file_bytes is not None else file_path.read_bytes()
        fname = file_name or (file_path.name if file_path else "file.bin")
        w(f"--{boundary}\r\n")
        w(f'Content-Disposition: form-data; name="{file_field}"; filename="{fname}"\r\n')
        w("Content-Type: application/octet-stream\r\n\r\n")
        body.write(data)
        w("\r\n")
    w(f"--{boundary}--\r\n")

    req = urllib.request.Request(url, data=body.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        _raise_http_error(url, e)
    except urllib.error.URLError as e:
        raise SystemExit(f"[submit] ✗ 连不上后端 {url}:{e.reason}\n"
                         f"  请检查 --server 地址与网络连通性(须能访问主办方公布的后端)")


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        _raise_http_error(url, e)
    except urllib.error.URLError as e:
        raise SystemExit(f"[submit] ✗ 连不上后端 {url}:{e.reason}\n"
                         f"  请检查 --server 地址与网络连通性(须能访问主办方公布的后端)")


def _pack_code_dir(code_dir: Path, exclude: set[str] | None = None) -> bytes:
    """整目录打成 .tar.gz(bytes)。exclude=None 用代码包默认排除;打 ckpt 目录传 exclude=set()
    (不排除任何子目录,否则会误删 ckpt 内容)。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(code_dir.rglob("*")):
            rel = p.relative_to(code_dir).parts
            if exclude is None:
                # 默认两层:junk 任意层排;产物目录仅顶层排(不误删嵌套同名源码)
                if any(part in _EXCLUDE_ANYWHERE for part in rel):
                    continue
                if rel and rel[0] in _EXCLUDE_TOPLEVEL:
                    continue
            elif any(part in exclude for part in rel):
                # 显式传 exclude(如打 ckpt 目录传 set()=不排):沿用全层匹配语义
                continue
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(code_dir)))
    return buf.getvalue()


def _pack_ckpt(ckpt_file: Path, stats_file: Path) -> bytes:
    """把单个 ckpt + 其 dataset_stats.pkl 打成 .tar.gz(bytes),只含这两个文件。
    ckpt 一律以 arcname `policy_last.ckpt` 入档(评测固定加载该名)——所以选手交
    policy_best 还是 policy_last 都行:**交哪个就评哪个,无需手动改名**。

    **解引用**写成普通文件成员(读真实字节 + TarInfo):选手若 `--ckpt` 指向符号链接
    (如 policy_best.ckpt→real.ckpt),tarfile.add 会存成 symlink 成员而被评测机以
    "含符号链接成员" 拒收;这里读字节写普通文件,规避该拒收。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for src, arc in ((ckpt_file, "policy_last.ckpt"), (stats_file, "dataset_stats.pkl")):
            data = src.read_bytes()               # 跟随 symlink 读真实字节
            ti = tarfile.TarInfo(name=arc)
            ti.size = len(data)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))      # 普通文件成员(非 symlink)
    return buf.getvalue()


def _resolve_token(a) -> str:
    """稳健解析 token,容忍以 `-` 开头的 token(部分 url-safe token 以 `-`/`_` 起头)。

    优先级:--token-file > env TRONCAMP_TOKEN > --token。**推荐用 env 或 `--token-file`,
    或 `--token=VALUE` 的等号形式**——裸 `--token -abc` 会被 argparse 当成 flag 解析失败,
    故本函数提供免裸传的路径。
    """
    if getattr(a, "token_file", None):
        tok = Path(a.token_file).read_text(encoding="utf-8").strip()
        if tok:
            return tok
    env_tok = os.environ.get("TRONCAMP_TOKEN")
    if env_tok and env_tok.strip():
        return env_tok.strip()
    if a.token:
        return a.token.strip()
    raise SystemExit(
        "缺 token:用 env `TRONCAMP_TOKEN=...`、`--token-file PATH`,或 `--token=VALUE`"
        "(等号形式,避免裸 `--token -abc` 被 argparse 当 flag)。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TronCamp ACT 四任务套餐提交 CLI")
    ap.add_argument("--server", default=None,
                    help="后端地址。默认已内置官方 mani 入口 "
                         "(https://submit.troncamp-mani.limxdynamics.com),一般无需填;"
                         "可用此参数或 env TRONCAMP_SERVER 覆盖")
    # token 不设 required:可由 env TRONCAMP_TOKEN / --token-file 提供(容忍 `-` 开头 token)。
    ap.add_argument("--token", default=None,
                    help="队伍 token。**以 `-` 开头的 token 用 `--token=VALUE` 等号形式或 env/"
                         "--token-file**,勿裸传 `--token -abc`(会被 argparse 当 flag)。")
    ap.add_argument("--token-file", default=None,
                    help="从文件读 token(首选,彻底回避 argparse 的 `-` 前缀陷阱)")
    ap.add_argument("--track", required=True, choices=["T1", "T2", "T3", "T4"])
    ap.add_argument("--ckpt", required=True,
                    help="单个 ACT checkpoint 文件(.ckpt);CLI 自动附带同目录的 dataset_stats.pkl,"
                         "并把你交的这个 ckpt 作为 policy_last.ckpt 打包上传(交哪个评哪个,无需改名)")
    ap.add_argument("--code-dir", default=None,
                    help="T2/T3/T4:RoboTwin 根目录(含 policy/,即 external/robotwin_local),"
                         "打包上传;worker 评测遮蔽 <repo_root>/policy。重型平台目录"
                         "(assets/envs/description)与嵌套 act_ckpt 会被自动排除;T1 不需要")
    a = ap.parse_args(argv)

    token = _resolve_token(a)
    server = (a.server or os.environ.get("TRONCAMP_SERVER") or DEFAULT_SERVER).strip().rstrip("/")
    ckpt_file = Path(a.ckpt)
    if not ckpt_file.is_file():
        print(f"--ckpt 不是文件(应指向单个 .ckpt 文件): {ckpt_file}", file=sys.stderr)
        return 2
    stats_file = ckpt_file.parent / "dataset_stats.pkl"
    if not stats_file.is_file():
        print(f"同目录缺 dataset_stats.pkl(ACT 评测需该归一化统计,训练会与 ckpt 一并产出于同目录): {stats_file}",
              file=sys.stderr)
        return 2

    # T2/T3/T4 代码包必填:在上传(可能很大的)ckpt **之前**校验,缺/非目录则 fail-fast,
    # 不白传一次 ckpt(评审 #4)。
    if a.track != "T1":
        if not a.code_dir:
            print("T2/T3/T4 必须传 --code-dir(整库代码包)", file=sys.stderr)
            return 2
        if not Path(a.code_dir).is_dir():
            print(f"--code-dir 不是目录(整库代码包要目录): {a.code_dir}", file=sys.stderr)
            return 2

    # 1) 打包 {你的 ckpt → policy_last.ckpt, dataset_stats.pkl} → 上传(评测机解档回目录传
    #    run_act_eval --ckpt-dir)。只打这两个文件:小、无歧义(不再上传整目录的多个 ckpt)。
    try:
        ckpt_blob = _pack_ckpt(ckpt_file, stats_file)
    except OSError as e:
        print(f"读取 ckpt / dataset_stats.pkl 失败(权限/损坏?): {e}", file=sys.stderr)
        return 2
    up = _post_multipart(f"{server}/api/upload", {"token": token},
                         "file", None, ckpt_blob, "ckpt.tar.gz")
    print(f"[submit] ckpt 上传完成 sha256={up['sha256'][:12]}… size={up['size']}")

    payload = {"token": token, "track": a.track,
               "ckpt_url": up["file_url"], "ckpt_sha256": up["sha256"]}

    # 2) T2/T3/T4 打包代码包(SHA-pin:把后端返回的 sha256 一并提交,评测机解档前再校验)
    if a.track != "T1":
        blob = _pack_code_dir(Path(a.code_dir))
        code_up = _post_multipart(f"{server}/api/upload", {"token": token},
                                  "file", None, blob, "code.tar.gz")
        payload["code_url"] = code_up["file_url"]
        payload["code_sha256"] = code_up["sha256"]
        print(f"[submit] 代码包上传完成 sha256={code_up['sha256'][:12]}… size={code_up['size']}")

    # 3) 提交
    res = _post_json(f"{server}/api/submit", payload)
    print(f"[submit] 已排队: #{res['id']} {res['track']} (队 {res['team']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
