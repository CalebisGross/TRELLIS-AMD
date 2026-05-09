"""
Audit tools/raster_repro/results.jsonl for schema drift, malformed lines,
and inconsistencies between the machine log and the daily files in
experiments/raster/daily/.

Exits 0 on clean, 1 on warnings, 2 on hard errors.

Usage:
    python tools/raster_repro/check_results.py
    python tools/raster_repro/check_results.py --strict     # warnings -> errors
    python tools/raster_repro/check_results.py --jsonl path # custom log path
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REQUIRED_TOP = {"ts", "test_name", "config", "build", "result"}
REQUIRED_RESULT = {"status", "frames_completed", "elapsed_s", "error"}
OPTIONAL_RESULT = {"fault_addr", "first_frame_s", "return_code", "stderr_tail"}
REQUIRED_BUILD = {"git_sha", "rocm_version", "gpu_arch"}

VALID_STATUS = {"PASS", "CRASH", "MIXED", "INCONCLUSIVE"}

REPO_ROOT = Path(__file__).resolve().parents[2]


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def err(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def note(self, msg): self.info.append(msg)

    def print(self):
        for m in self.info:
            print(f"info: {m}")
        for m in self.warnings:
            print(f"warn: {m}")
        for m in self.errors:
            print(f"ERROR: {m}", file=sys.stderr)

    def exit_code(self, strict):
        if self.errors:
            return 2
        if self.warnings and strict:
            return 1
        return 0


def parse_test_num(name):
    m = re.match(r"^test(\d+)", name or "")
    return int(m.group(1)) if m else None


def audit_jsonl(path, report):
    if not path.exists():
        report.err(f"jsonl not found: {path}")
        return []

    entries = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                e = json.loads(stripped)
            except json.JSONDecodeError as ex:
                report.err(f"{path}:{lineno}: invalid JSON: {ex}")
                continue
            entries.append((lineno, e))

    seen_names = {}
    prev_ts = None
    for lineno, e in entries:
        missing = REQUIRED_TOP - set(e.keys())
        if missing:
            report.err(f"{path}:{lineno}: missing top-level keys {sorted(missing)}")
            continue

        result = e.get("result", {}) or {}
        missing_r = REQUIRED_RESULT - set(result.keys())
        if missing_r:
            report.err(f"{path}:{lineno}: result missing {sorted(missing_r)}")

        unknown_r = set(result.keys()) - (REQUIRED_RESULT | OPTIONAL_RESULT)
        if unknown_r:
            report.warn(f"{path}:{lineno}: unknown result keys {sorted(unknown_r)}")

        status = result.get("status")
        if status not in VALID_STATUS:
            report.err(f"{path}:{lineno}: invalid status {status!r}")

        build = e.get("build", {}) or {}
        missing_b = REQUIRED_BUILD - set(build.keys())
        if missing_b:
            report.warn(f"{path}:{lineno}: build missing {sorted(missing_b)}")

        # Timestamp ordering check.
        ts = e.get("ts")
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                report.warn(f"{path}:{lineno}: malformed ts {ts!r}")
            else:
                if prev_ts is not None and t < prev_ts:
                    report.warn(f"{path}:{lineno}: timestamp regressed (prev was newer)")
                prev_ts = t

        # Duplicate test name check.
        name = e.get("test_name")
        if name in seen_names:
            report.warn(f"{path}:{lineno}: test_name {name!r} duplicates line {seen_names[name]}")
        else:
            seen_names[name] = lineno

    report.note(f"{path.name}: {len(entries)} entries, {len(seen_names)} unique test names")
    return entries


def audit_daily_consistency(entries, report):
    daily_dir = REPO_ROOT / "experiments" / "raster" / "daily"
    if not daily_dir.exists():
        report.warn(f"daily dir missing: {daily_dir}")
        return

    # Group entries by date (from ts).
    by_date = {}
    for _, e in entries:
        ts = e.get("ts", "")
        date = ts[:10] if ts else None
        if not date:
            continue
        by_date.setdefault(date, []).append(e)

    for date, day_entries in by_date.items():
        daily_file = daily_dir / f"{date}.md"
        if not daily_file.exists():
            report.warn(f"jsonl has {len(day_entries)} entries on {date} but {daily_file.name} doesn't exist")
            continue

        text = daily_file.read_text()
        for e in day_entries:
            num = parse_test_num(e.get("test_name", ""))
            if num is None:
                continue
            row_re = re.compile(rf"^\|\s*{num}\s*\|", re.MULTILINE)
            if not row_re.search(text):
                report.warn(f"test {num} in jsonl on {date} but no row in {daily_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",
                        default=str(REPO_ROOT / "tools" / "raster_repro" / "results.jsonl"),
                        help="path to results.jsonl")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="skip the daily-file cross-check")
    args = parser.parse_args()

    report = Report()
    entries = audit_jsonl(Path(args.jsonl), report)
    if not args.no_cross_check and entries:
        audit_daily_consistency(entries, report)

    report.print()
    return report.exit_code(args.strict)


if __name__ == "__main__":
    sys.exit(main())
