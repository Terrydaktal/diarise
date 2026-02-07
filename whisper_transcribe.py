#!/usr/bin/env python3
"""
Transcribe audio with Whisper "base" (or other sizes) using faster-whisper.

Outputs:
  - TXT transcript
  - JSON with segment timestamps

Example:
  . .venv/bin/activate
  python whisper_transcribe.py Recording_3487_condensed.m4a --model base --out-prefix Recording_3487_condensed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def configure_hf_cache() -> None:
    # HuggingFace Hub defaults to ~/.cache/huggingface; in some environments this path can be
    # non-writable (e.g. created by another user/root). Use a repo-local cache by default.
    if "HF_HOME" not in os.environ:
        repo_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
        os.environ["HF_HOME"] = repo_cache

def ffprobe_duration_seconds(path: str) -> float:
    try:
        cp = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffprobe failed: {e}") from e
    s = (cp.stdout or "").strip()
    return float(s) if s else 0.0

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

def get_file_anchor_time(path: str) -> tuple[datetime, str]:
    """
    Returns (anchor_datetime, source) where source is "birth" or "mtime".
    """
    try:
        birth = subprocess.check_output(["stat", "-c", "%w", path], text=True).strip()
    except Exception as e:
        raise RuntimeError(f"stat failed for {path}: {e}") from e

    if birth and birth != "-":
        return _parse_stat_time(birth), "birth"

    mtime = subprocess.check_output(["stat", "-c", "%y", path], text=True).strip()
    return _parse_stat_time(mtime), "mtime"

def load_diarise_anchor_file(report_json_path: str) -> str:
    try:
        with open(report_json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as e:
        die(f"Failed to read diarise report JSON: {e}")
    p = j.get("input")
    if not isinstance(p, str) or not p:
        die("Diarise report JSON missing 'input' path")
    return p

def guess_diarise_report_for_input(input_path: str) -> str | None:
    """
    Best-effort: if INPUT looks like an output from diarise, try to find the matching
    `*_report.json` next to it and use its `input` as the anchor file.
    """
    d = os.path.dirname(os.path.abspath(input_path)) or "."
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Common: <stem>_condensed(.ext) and report is <stem>_report.json
    if "_condensed" in base:
        stem = base.split("_condensed", 1)[0]
        cand = os.path.join(d, f"{stem}_report.json")
        if os.path.isfile(cand):
            return cand
    return None

def write_sectioned_transcript(
    *,
    out_path: str,
    anchor_file: str,
    segments: list[dict[str, Any]],
    bin_minutes: int,
) -> None:
    if bin_minutes <= 0:
        return

    anchor_dt, anchor_src = get_file_anchor_time(anchor_file)
    bin_s = float(bin_minutes) * 60.0

    # Group by bin index using segment start.
    bins: dict[int, list[dict[str, Any]]] = {}
    max_i = 0
    for s in segments:
        try:
            start = float(s["start"])
        except Exception:
            continue
        i = int(start // bin_s) if start > 0 else 0
        max_i = max(max_i, i)
        bins.setdefault(i, []).append(s)

    def fmt_dt(dt: datetime) -> str:
        return dt.isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append(f"# Transcript sections ({bin_minutes} min bins)")
    lines.append(f"# anchor_file: {os.path.abspath(anchor_file)}")
    lines.append(f"# anchor_time: {fmt_dt(anchor_dt)} ({anchor_src})")
    lines.append("")

    for i in range(0, max_i + 1):
        bin_start_s = i * bin_s
        bin_end_s = (i + 1) * bin_s
        abs_start = anchor_dt + timedelta(seconds=bin_start_s)
        abs_end = anchor_dt + timedelta(seconds=bin_end_s)
        lines.append(
            f"## {fmt_dt(abs_start)} to {fmt_dt(abs_end)}  (t={bin_start_s/3600:.2f}h to {bin_end_s/3600:.2f}h)"
        )
        seg_list = sorted(bins.get(i, []), key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))))
        if not seg_list:
            lines.append("[no transcript in this window]")
            lines.append("")
            continue
        for s in seg_list:
            start = float(s["start"])
            end = float(s["end"])
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            lines.append(f"[{start:8.2f}s -> {end:8.2f}s] {text}")
        lines.append("")

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main() -> None:
    configure_hf_cache()

    ap = argparse.ArgumentParser(description="Transcribe audio using Whisper via faster-whisper.")
    ap.add_argument("input", help="Input audio/video file (any format ffmpeg can read)")
    ap.add_argument("--model", default="base", help="Whisper model size/name (default: base)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device for inference (default: auto)")
    ap.add_argument("--compute-type", default="auto",
                    help="CTranslate2 compute type, e.g. auto, int8, int8_float16, float16, float32 (default: auto)")
    ap.add_argument("--language", default=None, help="Language code, e.g. en. Default: auto-detect")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task (default: transcribe)")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    ap.add_argument("--vad-filter", action="store_true", help="Enable faster-whisper's internal VAD filter")
    ap.add_argument("--log-every", type=float, default=300.0, help="Progress log interval seconds (default: 300)")
    ap.add_argument("--section-minutes", type=int, default=30,
                    help="Also write a sectioned transcript (*_sectioned.txt) with this bin size (default: 30). Set 0 to disable.")
    ap.add_argument("--anchor-file", default=None,
                    help="File whose filesystem time anchors the section timestamps (default: INPUT).")
    ap.add_argument("--diarise-report", default=None,
                    help="Path to diarise JSON report; uses its 'input' as the anchor file (overridden by --anchor-file).")
    ap.add_argument("--out-prefix", default=None,
                    help="Output prefix (default: INPUT basename without extension)")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        die(f"Input file not found: {in_path}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die("Missing dependency: faster-whisper. Install with: pip install faster-whisper")

    out_prefix = args.out_prefix
    if not out_prefix:
        out_prefix = os.path.splitext(os.path.basename(in_path))[0]

    out_txt = f"{out_prefix}.txt"
    out_json = f"{out_prefix}.json"
    out_sectioned = f"{out_prefix}_sectioned.txt"

    total_s = 0.0
    try:
        total_s = ffprobe_duration_seconds(in_path)
    except Exception:
        total_s = 0.0

    # Device selection: "auto" lets ctranslate2 choose, but we also want a clear message.
    device = args.device
    if device == "auto":
        # Prefer GPU if it looks usable; otherwise CPU.
        try:
            import ctypes

            ctypes.CDLL("libcuda.so.1")  # quick sanity check; doesn't guarantee working GPU
            device = "cuda"
        except Exception:
            device = "cpu"

    print(f"[info] model={args.model} device={device} compute_type={args.compute_type}", file=sys.stderr)
    if device == "cuda":
        print("[info] If this errors, your NVIDIA driver/userspace may be mismatched; re-run with --device cpu.", file=sys.stderr)

    try:
        model = WhisperModel(args.model, device=device, compute_type=args.compute_type)
    except Exception as e:
        if device == "cuda":
            die(f"Failed to initialize CUDA backend ({e}). Try: --device cpu")
        raise

    t0 = time.time()
    segments, info = model.transcribe(
        in_path,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
    )

    segs_out: List[Dict[str, Any]] = []
    texts: List[str] = []
    next_log_at = args.log_every if args.log_every > 0 else float("inf")
    last_end = 0.0
    for s in segments:
        last_end = float(s.end)
        if last_end >= next_log_at:
            if total_s > 0:
                pct = max(0.0, min(100.0, (last_end / total_s) * 100.0))
                print(f"[progress] {last_end/3600:.2f}h / {total_s/3600:.2f}h ({pct:.1f}%)", file=sys.stderr)
            else:
                print(f"[progress] {last_end/3600:.2f}h", file=sys.stderr)
            next_log_at += args.log_every

        segs_out.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
            }
        )
        texts.append(s.text.strip())

    ensure_parent_dir(out_txt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join([t for t in texts if t]) + "\n")

    ensure_parent_dir(out_json)
    payload: Dict[str, Any] = {
        "input": os.path.abspath(in_path),
        "model": args.model,
        "device": device,
        "compute_type": args.compute_type,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": segs_out,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    try:
        anchor_file = in_path
        if args.anchor_file:
            anchor_file = args.anchor_file
        elif args.diarise_report:
            anchor_file = load_diarise_anchor_file(args.diarise_report)
        else:
            guessed = guess_diarise_report_for_input(in_path)
            if guessed:
                anchor_file = load_diarise_anchor_file(guessed)

        write_sectioned_transcript(
            out_path=out_sectioned,
            anchor_file=anchor_file,
            segments=segs_out,
            bin_minutes=args.section_minutes,
        )
    except Exception as e:
        print(f"[warn] failed to write sectioned transcript: {e}", file=sys.stderr)

    dt = time.time() - t0
    if args.section_minutes > 0:
        print(f"[done] wrote {out_txt}, {out_json}, {out_sectioned} in {dt/60:.1f} min", file=sys.stderr)
    else:
        print(f"[done] wrote {out_txt} and {out_json} in {dt/60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
