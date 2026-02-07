#!/usr/bin/env python3
"""
Group a faster-whisper JSON transcript into fixed time bins (default 30 minutes),
and label each bin with an absolute timestamp derived from a file's creation time.

Notes:
- Uses filesystem "Birth" time if available (`stat -c %w`), otherwise mtime (`stat -c %y`).
- Absolute timestamps are anchored to the provided `--source-file`. For condensed audio,
  these are "real" wall-clock labels only in the sense of that anchor; they do not map
  back to the original uncondensed recording timeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


def die(msg: str, code: int = 1) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _parse_stat_time(s: str) -> datetime:
    # Example: "2026-02-06 21:52:36.389693649 +0000"
    s = s.strip()
    if s == "-" or not s:
        raise ValueError("empty/unknown time")

    parts = s.split()
    if len(parts) < 3:
        raise ValueError(f"unexpected time format: {s}")
    date_s, time_s, tz_s = parts[0], parts[1], parts[2]

    # Trim ns -> us for Python.
    if "." in time_s:
        hhmmss, frac = time_s.split(".", 1)
        frac = (frac + "000000")[:6]
        time_s = f"{hhmmss}.{frac}"

    # Convert +0000 -> +00:00 for fromisoformat.
    if len(tz_s) == 5 and (tz_s[0] in "+-") and tz_s[1:].isdigit():
        tz_s = f"{tz_s[0]}{tz_s[1:3]}:{tz_s[3:5]}"

    return datetime.fromisoformat(f"{date_s}T{time_s}{tz_s}")


def get_file_anchor_time(path: str) -> Tuple[datetime, str]:
    """
    Returns (anchor_datetime, source) where source indicates which stat field was selected.
    """
    try:
        fields = [
            ("access", "%x"),
            ("modify", "%y"),
            ("change", "%z"),
            ("birth", "%w"),
        ]
        parsed: List[Tuple[datetime, str]] = []
        for name, fmt in fields:
            s = subprocess.check_output(["stat", "-c", fmt, path], text=True).strip()
            if s and s != "-":
                parsed.append((_parse_stat_time(s), name))
    except Exception as e:
        raise RuntimeError(f"stat failed for {path}: {e}") from e

    if not parsed:
        raise RuntimeError(f"no usable stat times for {path}")
    dt, src = min(parsed, key=lambda x: x[0])
    return dt, f"earliest:{src}"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def load_segments(whisper_json_path: str) -> List[Segment]:
    with open(whisper_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    segs = j.get("segments")
    if not isinstance(segs, list):
        die(f"{whisper_json_path} missing 'segments' list")

    out: List[Segment] = []
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            continue
        try:
            start = float(s["start"])
            end = float(s["end"])
            text = str(s.get("text", "")).strip()
        except Exception as e:
            raise RuntimeError(f"bad segment {i}: {e}") from e
        if end > start and text:
            out.append(Segment(start=start, end=end, text=text))
    return out


def format_dt(dt: datetime) -> str:
    # Stable, grep-friendly timestamp.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def bin_index(t: float, bin_s: float) -> int:
    if t <= 0:
        return 0
    return int(t // bin_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Section a Whisper JSON transcript into fixed time windows.")
    ap.add_argument("--whisper-json", required=True, help="Input JSON from whisper_transcribe.py")
    ap.add_argument("--source-file", required=True, help="File whose filesystem time anchors the absolute timestamps")
    ap.add_argument("--bin-minutes", type=int, default=30, help="Section size in minutes (default: 30)")
    ap.add_argument("--out", required=True, help="Output sectioned transcript text file")
    args = ap.parse_args()

    anchor_dt, anchor_src = get_file_anchor_time(args.source_file)
    segs = load_segments(args.whisper_json)

    bin_s = float(args.bin_minutes) * 60.0
    if bin_s <= 0:
        die("--bin-minutes must be > 0")

    # Group segments by bin.
    bins: Dict[int, List[Segment]] = {}
    for s in segs:
        i = bin_index(s.start, bin_s)
        bins.setdefault(i, []).append(s)

    if not bins:
        die("No non-empty segments found in JSON")

    max_i = max(bins.keys())

    lines: List[str] = []
    lines.append(f"# Transcript sections ({args.bin_minutes} min bins)")
    lines.append(f"# anchor_file: {args.source_file}")
    lines.append(f"# anchor_time: {format_dt(anchor_dt)} ({anchor_src})")
    lines.append("")

    for i in range(0, max_i + 1):
        bin_start_s = i * bin_s
        bin_end_s = (i + 1) * bin_s
        abs_start = anchor_dt + timedelta(seconds=bin_start_s)
        abs_end = anchor_dt + timedelta(seconds=bin_end_s)

        title = f"## {format_dt(abs_start)} to {format_dt(abs_end)}  (t={bin_start_s/3600:.2f}h to {bin_end_s/3600:.2f}h)"
        lines.append(title)

        seg_list = bins.get(i, [])
        if not seg_list:
            lines.append("[no transcript in this window]")
            lines.append("")
            continue

        # Keep in original order.
        seg_list = sorted(seg_list, key=lambda s: (s.start, s.end))
        for s in seg_list:
            rel = f"[{s.start:8.2f}s -> {s.end:8.2f}s]"
            lines.append(f"{rel} {s.text}")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
