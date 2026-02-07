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
from bisect import bisect_right
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

def _whisper_cpu_compute_type(compute_type: str) -> str:
    ct = (compute_type or "").lower()
    if "float16" in ct or ct == "float16":
        return "int8"
    return compute_type

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
    Returns (anchor_datetime, source) where source indicates which stat field was selected.
    """
    try:
        fields = [
            ("access", "%x"),
            ("modify", "%y"),
            ("change", "%z"),
            ("birth", "%w"),
        ]
        parsed: list[tuple[datetime, str]] = []
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

def guess_prevad_report_for_input(input_path: str) -> str | None:
    """
    If INPUT looks like a condensed output from --prevad preprocessing, try to find
    the matching `<stem>_prevad.json` next to it.
    """
    d = os.path.dirname(os.path.abspath(input_path)) or "."
    base = os.path.splitext(os.path.basename(input_path))[0]

    if base.endswith("_prevad_condensed"):
        stem = base[: -len("_prevad_condensed")]
        cand = os.path.join(d, f"{stem}_prevad.json")
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

def write_timestamped_transcript(
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
    lines.append(f"# Transcript (timestamped, {bin_minutes} min bins)")
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
            abs_s = anchor_dt + timedelta(seconds=start)
            abs_e = anchor_dt + timedelta(seconds=end)
            lines.append(
                f"[{fmt_dt(abs_s)} -> {fmt_dt(abs_e)}] [{start:8.2f}s -> {end:8.2f}s] {text}"
            )
        lines.append("")

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def merge_intervals(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    out: list[tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out

def clamp_intervals(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in intervals:
        s2 = max(lo, s)
        e2 = min(hi, e)
        if e2 > s2:
            out.append((s2, e2))
    return out

def drop_short(intervals: list[tuple[float, float]], min_len: float) -> list[tuple[float, float]]:
    return [(s, e) for s, e in intervals if (e - s) >= min_len]

def webrtc_vad_clip_timestamps(
    *,
    input_path: str,
    dur_s: float,
    sr: int = 16000,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    min_silence_ms: int = 300,
    min_speech_s: float = 0.20,
    pad_pre_s: float = 0.35,
    pad_post_s: float = 0.60,
    gap_cut_s: float = 10.0,
    log_every_s: float = 300.0,
) -> list[float]:
    """
    Returns clip_timestamps as a flat list: [start0, end0, start1, end1, ...]
    for use with faster-whisper's `clip_timestamps`, in seconds in the ORIGINAL timeline.
    """
    try:
        import webrtcvad
    except ImportError:
        die("Missing dependency: webrtcvad. Install with: pip install webrtcvad")

    if frame_ms not in (10, 20, 30):
        die("--prevad-frame-ms must be one of 10, 20, 30")
    if not (0 <= aggressiveness <= 3):
        die("--prevad-aggressiveness must be 0..3")

    vad = webrtcvad.Vad(aggressiveness)
    frame_s = frame_ms / 1000.0
    samples_per_frame = int(sr * frame_s)
    bytes_per_frame = samples_per_frame * 2  # s16le mono
    min_silence_frames = max(1, int(round((min_silence_ms / 1000.0) / frame_s)))

    # Stream decode to PCM. Don't pipe stderr without draining it.
    # On corrupted/dirty inputs ffmpeg can spam stderr and block if we don't drain it.
    ffmpeg_stderr_f = None
    ff_cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-"
    ]
    try:
        import tempfile

        ffmpeg_stderr_f = tempfile.NamedTemporaryFile(prefix="webrtc_vad_ffmpeg_", suffix=".err", delete=False)
        proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ffmpeg_stderr_f)
    except Exception:
        proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.stdout is None:
        die("Failed to start ffmpeg for WebRTC VAD PCM streaming")

    segments: list[tuple[float, float]] = []
    in_speech = False
    speech_start = 0.0
    silence_run = 0
    processed_frames = 0
    next_log_at = log_every_s if log_every_s > 0 else float("inf")

    while True:
        buf = proc.stdout.read(bytes_per_frame)
        if not buf:
            break
        if len(buf) < bytes_per_frame:
            buf = buf + b"\x00" * (bytes_per_frame - len(buf))

        t_end = (processed_frames + 1) * frame_s
        if t_end >= next_log_at:
            total = dur_s if dur_s > 0 else t_end
            print(f"[prevad] processed {t_end/3600:.2f}h / {total/3600:.2f}h", file=sys.stderr)
            next_log_at += log_every_s

        is_speech = vad.is_speech(buf, sr)
        if is_speech:
            if not in_speech:
                in_speech = True
                speech_start = processed_frames * frame_s
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    end_t = (processed_frames + 1 - silence_run) * frame_s
                    segments.append((speech_start, end_t))
                    in_speech = False
                    silence_run = 0

        processed_frames += 1

    total_dur = min(float(dur_s), processed_frames * frame_s) if dur_s else (processed_frames * frame_s)
    if in_speech:
        segments.append((speech_start, total_dur))

    proc.stdout.close()
    rc = proc.wait()
    if ffmpeg_stderr_f is not None:
        try:
            ffmpeg_stderr_f.close()
        except Exception:
            pass
    if rc != 0:
        tail = ""
        if ffmpeg_stderr_f is not None:
            try:
                with open(ffmpeg_stderr_f.name, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()[-20:]
                tail = "".join(lines).strip()
            except Exception:
                tail = ""
        if tail:
            die(f"ffmpeg WebRTC PCM streaming failed (exit {rc}); stderr tail:\n{tail}")
        die(f"ffmpeg WebRTC PCM streaming failed (exit {rc})")

    # Postprocess: clamp, drop shorts, pad, merge across short pauses.
    segments = clamp_intervals(segments, 0.0, total_dur)
    segments = drop_short(segments, min_speech_s)

    padded: list[tuple[float, float]] = []
    for s, e in segments:
        padded.append((s - pad_pre_s, e + pad_post_s))
    padded = clamp_intervals(padded, 0.0, total_dur)
    merged = merge_intervals(padded, gap=float(gap_cut_s))
    merged = drop_short(merged, min_speech_s)

    clip: list[float] = []
    for s, e in merged:
        clip.extend([float(s), float(e)])
    return clip


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _pairwise_clip_to_intervals(clip_timestamps: list[float]) -> list[list[float]]:
    if len(clip_timestamps) % 2 != 0:
        die("Internal error: clip_timestamps must have even length")
    out: list[list[float]] = []
    for i in range(0, len(clip_timestamps), 2):
        s = float(clip_timestamps[i])
        e = float(clip_timestamps[i + 1])
        if e > s:
            out.append([s, e])
    return out


def _prevad_write_report_json(report_path: str, *, input_path: str, vad_keep: list[list[float]], settings: dict[str, Any]) -> None:
    ensure_parent_dir(report_path)
    chunks: list[dict[str, float]] = []
    t = 0.0
    for s, e in vad_keep:
        s2 = float(s)
        e2 = float(e)
        if e2 <= s2:
            continue
        dur = e2 - s2
        chunks.append(
            {
                "orig_start": s2,
                "orig_end": e2,
                "condensed_start": t,
                "condensed_end": t + dur,
            }
        )
        t += dur

    payload = {
        "input": os.path.abspath(input_path),
        "vad_keep": vad_keep,
        "settings": settings,
        "timeline_map": {
            "type": "stitch",
            "chunks": chunks,
            "condensed_duration_s": t,
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _load_timeline_map(report_json_path: str) -> tuple[str | None, list[dict[str, float]]]:
    try:
        with open(report_json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as e:
        die(f"Failed to read timeline map JSON: {e}")

    orig = j.get("input")
    if orig is not None and not isinstance(orig, str):
        orig = None

    tm = j.get("timeline_map")
    if not isinstance(tm, dict):
        die("Timeline map JSON missing 'timeline_map' object")
    chunks = tm.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        die("Timeline map JSON missing non-empty 'timeline_map.chunks' list")

    out_chunks: list[dict[str, float]] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        try:
            out_chunks.append(
                {
                    "orig_start": float(c["orig_start"]),
                    "orig_end": float(c["orig_end"]),
                    "condensed_start": float(c["condensed_start"]),
                    "condensed_end": float(c["condensed_end"]),
                }
            )
        except Exception:
            continue
    if not out_chunks:
        die("Timeline map JSON had no valid chunks")
    out_chunks.sort(key=lambda x: (x["condensed_start"], x["condensed_end"]))
    return orig, out_chunks

def _map_condensed_time_to_orig(t: float, chunks: list[dict[str, float]], starts: list[float]) -> float:
    if not chunks:
        return t
    i = bisect_right(starts, t) - 1
    if i < 0:
        i = 0
    if i >= len(chunks):
        i = len(chunks) - 1

    c = chunks[i]
    # If t lands just after the end (rounding), move forward.
    if t > c["condensed_end"] and i + 1 < len(chunks):
        c = chunks[i + 1]

    offset = t - c["condensed_start"]
    orig_len = max(0.0, c["orig_end"] - c["orig_start"])
    offset = max(0.0, min(offset, orig_len))
    return c["orig_start"] + offset

def _map_segments_condensed_to_orig(segments: list[dict[str, Any]], chunks: list[dict[str, float]]) -> list[dict[str, Any]]:
    starts = [c["condensed_start"] for c in chunks]
    out: list[dict[str, Any]] = []
    for s in segments:
        try:
            cs = float(s["start"])
            ce = float(s["end"])
        except Exception:
            out.append(s)
            continue
        ns = _map_condensed_time_to_orig(cs, chunks, starts)
        ne = _map_condensed_time_to_orig(ce, chunks, starts)
        s2 = dict(s)
        s2["start"] = float(ns)
        s2["end"] = float(ne)
        out.append(s2)
    return out


def _prevad_render_condensed_audio(
    *,
    diarise_path: str,
    input_path: str,
    report_json_path: str,
    condensed_out: str,
) -> None:
    # Use the current interpreter so diarise sees the same venv deps.
    cmd = [
        sys.executable,
        diarise_path,
        input_path,
        condensed_out,
        "--condense-from-json",
        report_json_path,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    configure_hf_cache()

    ap = argparse.ArgumentParser(description="Transcribe audio using Whisper via faster-whisper.")
    ap.add_argument("input", help="Input audio/video file (any format ffmpeg can read)")
    ap.add_argument("--model", default="base", help="Whisper model size/name (default: base)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device for inference (default: auto)")
    ap.add_argument("--compute-type", default="float16",
                    help="CTranslate2 compute type, e.g. int8, int8_float16, float16, float32 (default: float16)")
    ap.add_argument("--language", default="en", help="Language code, e.g. en (default: en)")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task (default: transcribe)")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    ap.set_defaults(vad_filter=True)
    ap.add_argument("--vad-filter", dest="vad_filter", action="store_true",
                    help="Enable faster-whisper's internal VAD filter (default: enabled)")
    ap.add_argument("--no-vad-filter", dest="vad_filter", action="store_false",
                    help="Disable faster-whisper's internal VAD filter")
    ap.add_argument("--prevad", action="store_true",
                    help="Run WebRTC VAD first and pass intervals to Whisper via clip_timestamps (disables built-in VAD).")
    ap.add_argument("--prevad-gap-cut", type=float, default=10.0,
                    help="In --prevad mode, merge speech across gaps <= this many seconds (default: 10)")
    ap.add_argument("--prevad-pad-pre", type=float, default=0.35, help="In --prevad mode, pad before speech (default: 0.35s)")
    ap.add_argument("--prevad-pad-post", type=float, default=0.60, help="In --prevad mode, pad after speech (default: 0.60s)")
    ap.add_argument("--prevad-min-speech", type=float, default=0.20, help="In --prevad mode, drop speech segments shorter than this (default: 0.20s)")
    ap.add_argument("--prevad-aggressiveness", type=int, default=2, help="In --prevad mode, WebRTC VAD aggressiveness 0..3 (default: 2)")
    ap.add_argument("--prevad-frame-ms", type=int, default=30, choices=[10, 20, 30], help="In --prevad mode, WebRTC VAD frame size ms (default: 30)")
    ap.add_argument("--prevad-min-silence-ms", type=int, default=300,
                    help="In --prevad mode, close speech after this much non-speech (default: 300ms)")
    ap.add_argument("--prevad-report-json", default=None,
                    help="In --prevad mode, also write a VAD keep-interval JSON (default: <out_prefix>_prevad.json)")
    ap.add_argument("--prevad-condensed-out", default=None,
                    help="In --prevad mode, also render a condensed audio file using diarise (default: <out_prefix>_prevad_condensed.m4a)")
    ap.add_argument("--prevad-no-preprocess", action="store_true",
                    help="In --prevad mode, skip writing the VAD JSON / condensed audio (transcription still uses clip_timestamps).")
    ap.add_argument("--timeline-map", default=None,
                    help="Map timestamps from a condensed stitched file back to original time using a prevad JSON report (with timeline_map).")
    ap.add_argument("--log-every", type=float, default=300.0, help="Progress log interval seconds (default: 300)")
    ap.add_argument("--section-minutes", type=int, default=30,
                    help="Also write *_sectioned.txt and *_timestamped.txt with this bin size (default: 30). Set 0 to disable.")
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
    out_timestamped = f"{out_prefix}_timestamped.txt"

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

    compute_type = args.compute_type
    if device == "cpu":
        compute_type2 = _whisper_cpu_compute_type(compute_type)
        if compute_type2 != compute_type:
            print(f"[info] cpu compute_type override: {compute_type} -> {compute_type2}", file=sys.stderr)
            compute_type = compute_type2
    print(f"[info] model={args.model} device={device} compute_type={compute_type}", file=sys.stderr)
    if device == "cuda":
        print("[info] If this errors, your NVIDIA driver/userspace may be mismatched; re-run with --device cpu.", file=sys.stderr)

    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            print(f"[warn] CUDA init failed ({e}); falling back to CPU", file=sys.stderr)
            device = "cpu"
            compute_type = _whisper_cpu_compute_type(compute_type)
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        else:
            raise

    # Optional condensed->original timestamp mapping (for transcribing stitched outputs).
    timeline_map_path = args.timeline_map
    if not timeline_map_path:
        timeline_map_path = guess_prevad_report_for_input(in_path)
    map_orig_input: str | None = None
    map_chunks: list[dict[str, float]] | None = None
    if timeline_map_path:
        map_orig_input, map_chunks = _load_timeline_map(timeline_map_path)
        print(f"[info] using timeline map: {timeline_map_path}", file=sys.stderr)

    clip_timestamps: list[float] | str = "0"
    vad_filter = bool(args.vad_filter)
    prevad_report_json_path: str | None = None
    prevad_condensed_out_path: str | None = None
    if args.prevad:
        print("[prevad] running WebRTC VAD -> clip_timestamps", file=sys.stderr)
        clip_timestamps = webrtc_vad_clip_timestamps(
            input_path=in_path,
            dur_s=total_s,
            frame_ms=args.prevad_frame_ms,
            aggressiveness=args.prevad_aggressiveness,
            min_silence_ms=args.prevad_min_silence_ms,
            min_speech_s=args.prevad_min_speech,
            pad_pre_s=args.prevad_pad_pre,
            pad_post_s=args.prevad_pad_post,
            gap_cut_s=args.prevad_gap_cut,
            log_every_s=max(0.0, float(args.log_every)),
        )
        vad_filter = False  # ignored when clip_timestamps is used, but keep explicit.
        if not clip_timestamps:
            die("Pre-VAD produced no keep intervals; nothing to transcribe.")

        # Preprocess (only in --prevad): write intervals JSON and render condensed audio.
        if not args.prevad_no_preprocess:
            report_path = args.prevad_report_json or f"{out_prefix}_prevad.json"
            prevad_report_json_path = report_path

            condensed_out = args.prevad_condensed_out or f"{out_prefix}_prevad_condensed.m4a"
            prevad_condensed_out_path = condensed_out

            vad_keep = _pairwise_clip_to_intervals(clip_timestamps)
            settings = {
                "backend": "webrtc",
                "sr": 16000,
                "frame_ms": args.prevad_frame_ms,
                "aggressiveness": args.prevad_aggressiveness,
                "min_silence_ms": args.prevad_min_silence_ms,
                "min_speech_s": args.prevad_min_speech,
                "pad_pre_s": args.prevad_pad_pre,
                "pad_post_s": args.prevad_pad_post,
                "gap_cut_s": args.prevad_gap_cut,
            }
            _prevad_write_report_json(report_path, input_path=in_path, vad_keep=vad_keep, settings=settings)
            print(f"[prevad] wrote report: {report_path}", file=sys.stderr)

            diarise_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diarise")
            if os.path.isfile(diarise_path):
                try:
                    print(f"[prevad] rendering condensed audio: {condensed_out}", file=sys.stderr)
                    _prevad_render_condensed_audio(
                        diarise_path=diarise_path,
                        input_path=in_path,
                        report_json_path=report_path,
                        condensed_out=condensed_out,
                    )
                except Exception as e:
                    print(f"[warn] prevad preprocessing failed to render condensed audio: {e}", file=sys.stderr)
            else:
                print("[warn] prevad preprocessing skipped: diarise script not found next to whisper_transcribe.py", file=sys.stderr)

    t0 = time.time()
    segments, info = model.transcribe(
        in_path,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=vad_filter,
        clip_timestamps=clip_timestamps,
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

    # If we transcribed a condensed stitched file and have a map, remap segment timestamps to original time.
    # Keep the raw condensed segments in the JSON for debugging.
    segs_raw: List[Dict[str, Any]] = segs_out
    if map_chunks is not None:
        segs_out = _map_segments_condensed_to_orig(segs_raw, map_chunks)

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
        "prevad": bool(args.prevad),
        "prevad_no_preprocess": bool(args.prevad_no_preprocess),
        "prevad_report_json": os.path.abspath(prevad_report_json_path) if prevad_report_json_path else None,
        "prevad_condensed_out": os.path.abspath(prevad_condensed_out_path) if prevad_condensed_out_path else None,
        "timeline_map": os.path.abspath(timeline_map_path) if timeline_map_path else None,
        "timeline_map_input": os.path.abspath(map_orig_input) if map_orig_input else None,
        "segments_condensed": segs_raw if map_chunks is not None else None,
        "vad_filter": bool(args.vad_filter) if not args.prevad else False,
        "clip_timestamps": clip_timestamps if args.prevad else None,
        "segments": segs_out,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    try:
        anchor_file = in_path
        if args.anchor_file:
            anchor_file = args.anchor_file
        elif map_orig_input and os.path.isfile(map_orig_input):
            anchor_file = map_orig_input
        elif args.diarise_report:
            anchor_file = load_diarise_anchor_file(args.diarise_report)
        else:
            guessed = guess_diarise_report_for_input(in_path)
            if guessed:
                anchor_file = load_diarise_anchor_file(guessed)

        write_sectioned_transcript(out_path=out_sectioned, anchor_file=anchor_file, segments=segs_out, bin_minutes=args.section_minutes)
        write_timestamped_transcript(out_path=out_timestamped, anchor_file=anchor_file, segments=segs_out, bin_minutes=args.section_minutes)
    except Exception as e:
        print(f"[warn] failed to write sectioned/timestamped transcript: {e}", file=sys.stderr)

    dt = time.time() - t0
    if args.section_minutes > 0:
        print(f"[done] wrote {out_txt}, {out_json}, {out_sectioned}, {out_timestamped} in {dt/60:.1f} min", file=sys.stderr)
    else:
        print(f"[done] wrote {out_txt} and {out_json} in {dt/60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
