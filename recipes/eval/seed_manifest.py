#!/usr/bin/env python3
"""Build a structured seed manifest from RoboTwin ACT eval output.

The runtime recorder is intentionally log-based so we do not need to patch
upstream RoboTwin internals beyond the existing project-local eval progress
prints. It records both expert seed probes and accepted policy rollouts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LOOP_RE = re.compile(r"\[eval-policy\] loop now_seed=(\d+) succ_seed=(\d+)/(\d+)")
EXPERT_DONE_RE = re.compile(r"\[eval-policy\] expert done seed=(\d+) plan_success=(True|False)")
SUMMARY_RE = re.compile(r"^([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)$")
SUCCESS_RATE_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%\s*,\s*current seed:\s*(\d+)"
)
RESULT_PATH_RE = re.compile(r"Data has been saved to\s+(.+_result\.txt)\s*$")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class EvalSeedManifestRecorder:
    """Incrementally parse eval stdout and write `seed_manifest.jsonl`."""

    def __init__(self) -> None:
        self._buffer = ""
        self._last_success_count = 0
        self._last_summary: dict[str, str] = {}
        self._probe_order: list[int] = []
        self._probes: dict[int, dict[str, object]] = {}
        self._rollouts: list[dict[str, object]] = []
        self._result_file: str | None = None
        self._test_num: int | None = None

    def feed(self, text: str) -> None:
        self._buffer += text
        while True:
            newline = self._buffer.find("\n")
            carriage = self._buffer.find("\r")
            candidates = [idx for idx in (newline, carriage) if idx >= 0]
            if not candidates:
                break
            idx = min(candidates)
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1 :]
            self.process_line(line)

        # Step progress can be noisy. Keep enough tail for a split structured line
        # without retaining an entire long rollout transcript.
        if len(self._buffer) > 8192:
            self._buffer = self._buffer[-8192:]

    def flush(self) -> None:
        if self._buffer:
            self.process_line(self._buffer)
            self._buffer = ""

    def process_line(self, raw_line: str) -> None:
        line = strip_ansi(raw_line).strip()
        if not line:
            return

        match = LOOP_RE.search(line)
        if match:
            self._test_num = int(match.group(3))
            return

        match = EXPERT_DONE_RE.search(line)
        if match:
            seed = int(match.group(1))
            if seed not in self._probes:
                self._probe_order.append(seed)
            self._probes[seed] = {
                "seed": seed,
                "expert_plan_success": match.group(2) == "True",
            }
            return

        match = SUMMARY_RE.match(line)
        if match:
            self._last_summary = {
                "task_name": match.group(1).strip(),
                "policy_name": match.group(2).strip(),
                "task_config": match.group(3).strip(),
                "ckpt_setting": match.group(4).strip(),
            }
            return

        match = SUCCESS_RATE_RE.search(line)
        if match:
            success_count = int(match.group(1))
            evaluated_count = int(match.group(2))
            seed = int(match.group(4))
            policy_success = success_count > self._last_success_count
            self._last_success_count = success_count
            record = {
                "episode_index": evaluated_count - 1,
                "seed": seed,
                "expert_plan_success": bool(self._probes.get(seed, {}).get("expert_plan_success", True)),
                "expert_check_success": True,
                "accepted_for_policy": True,
                "policy_success": policy_success,
                "cumulative_success_count": success_count,
                "cumulative_evaluated_count": evaluated_count,
                "cumulative_success_rate": float(match.group(3)),
            }
            record.update(self._last_summary)
            self._rollouts.append(record)
            return

        match = RESULT_PATH_RE.search(line)
        if match:
            self._result_file = match.group(1)

    def write_manifest(
        self,
        robotwin_root: Path | None = None,
        eval_result_dir: Path | None = None,
        output_path: Path | None = None,
    ) -> Path | None:
        self.flush()
        target_dir = self._resolve_eval_result_dir(robotwin_root, eval_result_dir)
        if output_path is None:
            if target_dir is None:
                return None
            output_path = target_dir / "seed_manifest.jsonl"
        else:
            output_path = output_path.expanduser().resolve(strict=False)
            target_dir = target_dir or output_path.parent

        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = list(self.iter_records(target_dir))
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return output_path

    def iter_records(self, eval_result_dir: Path | None = None) -> list[dict[str, object]]:
        rollout_seeds = {int(record["seed"]) for record in self._rollouts}
        metadata: dict[str, object] = {
            "record_type": "metadata",
            "generated_at": utc_now(),
            "schema": "tron2_eval_seed_manifest.v1",
            "source": "recipes/eval/seed_manifest.py",
            "expert_probe_count": len(self._probe_order),
            "policy_rollout_count": len(self._rollouts),
            "policy_success_count": int(self._rollouts[-1]["cumulative_success_count"]) if self._rollouts else 0,
        }
        if self._test_num is not None:
            metadata["requested_test_num"] = self._test_num
        if self._result_file:
            metadata["result_file"] = self._result_file
        if eval_result_dir is not None:
            metadata["eval_result_dir"] = str(eval_result_dir)
        if self._rollouts:
            metadata.update(
                {
                    key: self._rollouts[-1][key]
                    for key in ("task_name", "policy_name", "task_config", "ckpt_setting")
                    if key in self._rollouts[-1]
                }
            )

        records = [metadata]
        for seed in self._probe_order:
            probe = dict(self._probes[seed])
            accepted = seed in rollout_seeds
            plan_success = bool(probe.get("expert_plan_success"))
            probe.update(
                {
                    "record_type": "expert_probe",
                    "accepted_for_policy": accepted,
                    "expert_check_success": accepted if plan_success else False,
                }
            )
            records.append(probe)

        for rollout in self._rollouts:
            record = dict(rollout)
            record["record_type"] = "policy_rollout"
            if eval_result_dir is not None:
                video = eval_result_dir / f"episode{record['episode_index']}.mp4"
                record["video_file"] = video.name
                if video.exists():
                    record["video_size_bytes"] = video.stat().st_size
            records.append(record)

        return records

    def _resolve_eval_result_dir(
        self,
        robotwin_root: Path | None,
        eval_result_dir: Path | None,
    ) -> Path | None:
        if eval_result_dir is not None:
            return eval_result_dir.expanduser().resolve(strict=False)
        if self._result_file is None:
            return None

        result_path = Path(self._result_file)
        if not result_path.is_absolute() and robotwin_root is not None:
            result_path = robotwin_root / result_path
        return result_path.expanduser().resolve(strict=False).parent


class TeeStream:
    """Forward writes to the original stream while feeding the manifest parser."""

    def __init__(self, wrapped: TextIO, recorder: EvalSeedManifestRecorder) -> None:
        self._wrapped = wrapped
        self._recorder = recorder

    def write(self, text: str) -> int:
        self._recorder.feed(text)
        return self._wrapped.write(text)

    def flush(self) -> None:
        self._wrapped.flush()

    def isatty(self) -> bool:
        return self._wrapped.isatty()

    @property
    def encoding(self) -> str | None:
        return self._wrapped.encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Eval stdout/stderr log to parse.")
    parser.add_argument("--robotwin-root", type=Path, default=None, help="Root used to resolve relative result paths.")
    parser.add_argument("--eval-result-dir", type=Path, default=None, help="Eval result directory containing videos.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path. Defaults to eval dir/seed_manifest.jsonl.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recorder = EvalSeedManifestRecorder()
    recorder.feed(args.log.read_text(encoding="utf-8", errors="replace"))
    path = recorder.write_manifest(
        robotwin_root=args.robotwin_root,
        eval_result_dir=args.eval_result_dir,
        output_path=args.output,
    )
    if path is None:
        print("No eval result directory found; pass --eval-result-dir or ensure the log has the result path.", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
