#!/usr/bin/env python3
"""
condense_day_audio.py

Pipeline:
  1) Pre-pass (cheap, streaming): detect "dead air" (very low peak+RMS) runs >= N seconds, remove them.
     - Uses windowed peak/RMS thresholds (conservative by default).
  2) Silero VAD pass (streaming): find speech segments on the pre-passed audio.
     - Keeps natural pauses by *only* removing non-speech gaps longer than N seconds.
  3) Renders a final condensed file + JSON reports (removed/kept intervals).

Requirements:
  - ffmpeg + ffprobe in PATH
  - Python: numpy, torch
  - Internet ONCE for torch.hub to fetch Silero VAD model (cached afterward)

Example:
  python3 condense_day_audio.py input.mp3 output.m4a

More conservative (miss less speech):
  python3 condense_day_audio.py input.mp3 output.m4a \
    --prepass-rms-db -65 --prepass-peak-db -60 \
    --vad-threshold 0.25 --pad-pre 0.40 --pad-post 0.70

Notes:
  - This script intentionally avoids "choppy" audio by NOT removing small pauses.
    It only removes dead-air gaps >= --gap-cut seconds (default 10).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable
from datetime import timedelta

import numpy as np


# -----------------------------
# Utility helpers
# -----------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def run(cmd: List[str], *, capture: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              text=text)
    except subprocess.CalledProcessError as e:
        if capture:
            stderr = (e.stderr or "").strip()
            stdout = (e.stdout or "").strip()
            detail = "\n".join([x for x in [stdout, stderr] if x])
            die(f"Command failed: {' '.join(cmd)}\n{detail}")
        die(f"Command failed: {' '.join(cmd)}")

def run_ffmpeg_with_progress(
    cmd: List[str],
    *,
    label: str,
    total_out_s: Optional[float] = None,
    emit_every_s: float = 10.0,
) -> None:
    """
    Run an ffmpeg command with `-progress pipe:2` enabled and periodically emit progress.
    This keeps long renders "alive" (useful in non-interactive environments) and gives the user
    a sense of where the command is.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if proc.stderr is None:
        die(f"{label}: failed to start ffmpeg")

    tail: deque[str] = deque(maxlen=60)
    out_time_s: Optional[float] = None
    last_emit = time.monotonic()

    def maybe_emit(force: bool = False) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if not force and (now - last_emit) < emit_every_s:
            return
        last_emit = now
        if out_time_s is None:
            print(f"[{label}] running...", file=sys.stderr, flush=True)
            return
        if total_out_s and total_out_s > 0:
            pct = max(0.0, min(100.0, (out_time_s / total_out_s) * 100.0))
            print(f"[{label}] {out_time_s/60:.1f}m / {total_out_s/60:.1f}m ({pct:.1f}%)", file=sys.stderr, flush=True)
        else:
            print(f"[{label}] {out_time_s/60:.1f}m", file=sys.stderr, flush=True)

    for raw in proc.stderr:
        line = raw.strip()
        if not line:
            continue
        tail.append(line)

        if line.startswith("out_time_ms="):
            try:
                out_time_s = int(line.split("=", 1)[1]) / 1_000_000.0
            except Exception:
                pass
        elif line.startswith("out_time="):
            # HH:MM:SS.micro
            v = line.split("=", 1)[1]
            try:
                hh, mm, ss = v.split(":")
                out_time_s = (int(hh) * 3600) + (int(mm) * 60) + float(ss)
            except Exception:
                pass
        elif line.startswith("progress="):
            # ffmpeg emits progress=continue repeatedly; throttle our own output.
            maybe_emit(force=False)

    rc = proc.wait()
    if rc != 0:
        detail = "\n".join(tail)
        die(f"{label}: ffmpeg failed (exit {rc}).\n{detail}")

def which_or_die(bin_name: str) -> None:
    if shutil.which(bin_name) is None:
        die(f"Missing dependency: {bin_name} not found in PATH")

def ffprobe_duration_seconds(path: str) -> float:
    # duration can be missing for some streams; this is the most common method.
    cp = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], capture=True)
    s = (cp.stdout or "").strip()
    if not s:
        die("ffprobe did not return duration (unsupported container/stream?)")
    try:
        return float(s)
    except ValueError:
        die(f"ffprobe returned non-numeric duration: {s}")

def ffprobe_audio_params(path: str) -> Tuple[int, int]:
    """
    Returns (sample_rate_hz, channels) for the first audio stream.
    """
    cp = run([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "json",
        path
    ], capture=True)
    try:
        j = json.loads(cp.stdout or "{}")
        streams = j.get("streams") or []
        if not streams:
            raise ValueError("no audio streams")
        s = streams[0]
        sr = int(s.get("sample_rate"))
        ch = int(s.get("channels"))
        return sr, ch
    except Exception as e:
        die(f"ffprobe could not read audio params: {e}")

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def configure_hf_cache() -> None:
    # HuggingFace Hub defaults to ~/.cache/huggingface; that can be non-writable on some setups.
    # Use a repo-local cache by default.
    if "HF_HOME" not in os.environ:
        repo_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
        os.environ["HF_HOME"] = repo_cache

def _parse_stat_time(s: str) -> datetime:
    # Example: "2026-02-06 21:06:36.909285246 +0000"
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

def ffmpeg_tail_decodes_ok(
    path: str,
    *,
    seconds_from_end: float = 60.0,
    probe_len_s: float = 10.0,
    timeout_s: float = 45.0,
) -> Tuple[bool, str]:
    """
    Quick corruption check: try decoding a short slice near the end of the file.
    Returns (ok, note). "ok" means we could decode without ffmpeg error output/exit.
    """
    seconds_from_end = max(1.0, float(seconds_from_end))
    probe_len_s = max(0.5, float(probe_len_s))

    # Use -sseof to seek from end. This is fast and doesn't require scanning the whole file.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-sseof",
        f"-{seconds_from_end}",
        "-t",
        f"{probe_len_s}",
        "-i",
        path,
        "-vn",
        "-f",
        "null",
        "-",
    ]
    timeout_s = max(float(timeout_s), float(probe_len_s) + 1.0)
    try:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"tail decode probe timed out after {timeout_s:.1f}s"
    err = (p.stderr or "").strip()
    if p.returncode != 0 or err:
        return False, "tail decode had ffmpeg errors (file may be truncated/corrupt)"
    return True, "tail decode ok"

def write_sectioned_transcript(
    *,
    out_path: str,
    anchor_file: str,
    segments: List[Dict[str, Any]],
    bin_minutes: int,
    tail_probe: Optional[Tuple[bool, str]] = None,
) -> None:
    if bin_minutes <= 0:
        return

    anchor_dt, anchor_src = get_file_anchor_time(anchor_file)
    # Interpret anchor_dt as the end of the recording, and back-calculate the start using ffprobe duration.
    dur_s = ffprobe_duration_seconds(anchor_file)
    start_dt = anchor_dt - timedelta(seconds=float(dur_s))
    if tail_probe is None:
        tail_ok, tail_note = ffmpeg_tail_decodes_ok(anchor_file)
    else:
        tail_ok, tail_note = tail_probe
    bin_s = float(bin_minutes) * 60.0

    # Group by bin index using segment start.
    bins: Dict[int, List[Dict[str, Any]]] = {}
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

    lines: List[str] = []
    lines.append(f"# Transcript sections ({bin_minutes} min bins)")
    lines.append(f"# anchor_file: {anchor_file}")
    lines.append(f"# anchor_time: {fmt_dt(anchor_dt)} ({anchor_src}) [assumed END of recording]")
    if not tail_ok:
        lines.append(f"# anchor_time_uncertain: true ({tail_note})")
    lines.append(f"# duration_s: {float(dur_s):.3f}")
    lines.append(f"# start_time: {fmt_dt(start_dt)} (anchor_time - duration)")
    lines.append("")

    for i in range(0, max_i + 1):
        bin_start_s = i * bin_s
        bin_end_s = (i + 1) * bin_s
        abs_start = start_dt + timedelta(seconds=bin_start_s)
        abs_end = start_dt + timedelta(seconds=bin_end_s)
        lines.append(
            f"## {fmt_dt(abs_start)} to {fmt_dt(abs_end)}  (t={bin_start_s/3600:.2f}h to {bin_end_s/3600:.2f}h)"
        )
        seg_list = sorted(bins.get(i, []), key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))))
        if not seg_list:
            lines.append("[no transcript in this window]")
            lines.append("")
            continue
        for s in seg_list:
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            lines.append(text)
        lines.append("")

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def write_timestamped_transcript(
    *,
    out_path: str,
    anchor_file: str,
    segments: List[Dict[str, Any]],
    bin_minutes: int,
    tail_probe: Optional[Tuple[bool, str]] = None,
) -> None:
    """
    Like write_sectioned_transcript, but each segment line includes absolute timestamps.
    """
    if bin_minutes <= 0:
        return

    anchor_dt, anchor_src = get_file_anchor_time(anchor_file)
    dur_s = ffprobe_duration_seconds(anchor_file)
    start_dt = anchor_dt - timedelta(seconds=float(dur_s))
    if tail_probe is None:
        tail_ok, tail_note = ffmpeg_tail_decodes_ok(anchor_file)
    else:
        tail_ok, tail_note = tail_probe
    bin_s = float(bin_minutes) * 60.0

    bins: Dict[int, List[Dict[str, Any]]] = {}
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

    lines: List[str] = []
    lines.append(f"# Transcript (timestamped, {bin_minutes} min bins)")
    lines.append(f"# anchor_file: {anchor_file}")
    lines.append(f"# anchor_time: {fmt_dt(anchor_dt)} ({anchor_src}) [assumed END of recording]")
    if not tail_ok:
        lines.append(f"# anchor_time_uncertain: true ({tail_note})")
    lines.append(f"# duration_s: {float(dur_s):.3f}")
    lines.append(f"# start_time: {fmt_dt(start_dt)} (anchor_time - duration)")
    lines.append("")

    for i in range(0, max_i + 1):
        bin_start_s = i * bin_s
        bin_end_s = (i + 1) * bin_s
        abs_start = start_dt + timedelta(seconds=bin_start_s)
        abs_end = start_dt + timedelta(seconds=bin_end_s)
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
                f"[{start:8.2f}s -> {end:8.2f}s] {text}"
            )
        lines.append("")

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def _choose_whisper_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctypes

        ctypes.CDLL("libcuda.so.1")
        return "cuda"
    except Exception:
        return "cpu"

def _whisper_cpu_compute_type(compute_type: str) -> str:
    # float16 compute types are generally GPU-focused; if we fall back to CPU, int8 is the
    # safest default for speed/compatibility.
    ct = (compute_type or "").lower()
    if "float16" in ct or ct == "float16":
        return "int8"
    return compute_type

def _resolve_whisper_model_name(name: str) -> str:
    """
    Map friendly aliases to faster-whisper model names.
    """
    raw = (name or "").strip()
    key = raw.lower().replace("_", "-")
    aliases = {
        # "large" is ambiguous (v2 vs v3) in casual usage. Default to the best general model.
        "large": "large-v3",
        "largev3": "large-v3",
        "whisper-large-v3": "large-v3",
        "large-turbo": "large-v3-turbo",
        "largev3-turbo": "large-v3-turbo",
        "whisper-large-v3-turbo": "large-v3-turbo",
    }
    return aliases.get(key, raw)

def _safe_slug(s: str) -> str:
    # Keep output filenames sane when model names include slashes (HF repo IDs) or spaces.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "").strip()).strip("_") or "model"

SUPPORTED_BATCH_AUDIO_EXTS = {
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".mka",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
}


def _list_audio_files_in_dir(input_dir: str) -> List[str]:
    out: List[str] = []
    try:
        with os.scandir(input_dir) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                ext = os.path.splitext(ent.name)[1].lower()
                if ext in SUPPORTED_BATCH_AUDIO_EXTS:
                    out.append(ent.path)
    except OSError as e:
        die(f"Failed to scan input directory: {e}")
    out.sort(key=lambda p: os.path.basename(p).lower())
    return out


def _transcription_outputs_complete(out_prefix: str, section_minutes: int) -> bool:
    required = [
        f"{out_prefix}.txt",
        f"{out_prefix}.json",
    ]
    if section_minutes > 0:
        required.extend(
            [
                f"{out_prefix}_sectioned.txt",
                f"{out_prefix}_timestamped.txt",
            ]
        )
    for p in required:
        if not os.path.isfile(p):
            return False
        try:
            if os.path.getsize(p) <= 0:
                return False
        except OSError:
            return False
    return True


def _batch_out_prefix_for_source(source_path: str, out_dir: str, model_slug: str) -> str:
    """
    Stable per-file prefix for directory mode.
    Includes the source extension token to avoid stem collisions (e.g. clip.wav vs clip.mp3).
    """
    filename = os.path.basename(source_path)
    stem, ext = os.path.splitext(filename)
    ext_token = (ext[1:] if ext.startswith(".") else ext).lower()
    if ext_token:
        stem = f"{stem}_{ext_token}"
    return os.path.join(out_dir, f"{stem}_whisper_{model_slug}")

def _intervals_to_clip_timestamps(intervals: List[Interval]) -> List[float]:
    clip: List[float] = []
    for s, e in intervals:
        if e > s:
            clip.extend([float(s), float(e)])
    return clip

def _split_intervals_max_len(intervals: List[Interval], max_len_s: float) -> List[Interval]:
    if max_len_s <= 0:
        return intervals
    out: List[Interval] = []
    for s0, e0 in intervals:
        s = float(s0)
        e = float(e0)
        if e <= s:
            continue
        while (e - s) > max_len_s:
            out.append((s, s + max_len_s))
            s += max_len_s
        out.append((s, e))
    return out

def _clip_timestamps_dicts(intervals: List[Interval]) -> List[Dict[str, float]]:
    return [{"start": float(s), "end": float(e)} for s, e in intervals if e > s]

def _build_stitch_chunks(keep_intervals: List[Interval]) -> tuple[List[Dict[str, float]], float]:
    """
    Given intervals in the ORIGINAL timeline that are stitched back-to-back into a new file,
    build a mapping "chunks" list describing original<->stitched time.
    """
    chunks: List[Dict[str, float]] = []
    t = 0.0
    for s, e in keep_intervals:
        s2 = float(s)
        e2 = float(e)
        if e2 <= s2:
            continue
        dur = e2 - s2
        chunks.append(
            {
                "orig_start": s2,
                "orig_end": e2,
                "stitched_start": t,
                "stitched_end": t + dur,
            }
        )
        t += dur
    return chunks, t

def _map_stitched_interval_to_orig(interval: Interval, chunks: List[Dict[str, float]]) -> List[Interval]:
    """
    Map [start,end] in stitched time back into one or more ORIGINAL intervals.
    Splits at stitch boundaries so we never bridge across removed dead-air.
    """
    s, e = float(interval[0]), float(interval[1])
    if e <= s:
        return []
    out: List[Interval] = []
    for c in chunks:
        cs = c["stitched_start"]
        ce = c["stitched_end"]
        if ce <= s:
            continue
        if cs >= e:
            break
        seg_s = max(s, cs)
        seg_e = min(e, ce)
        if seg_e <= seg_s:
            continue
        # Convert offset within chunk to original.
        off_s = seg_s - cs
        off_e = seg_e - cs
        out.append((c["orig_start"] + off_s, c["orig_start"] + off_e))
    return out

def _map_stitched_intervals_to_orig(intervals: List[Interval], chunks: List[Dict[str, float]]) -> List[Interval]:
    out: List[Interval] = []
    for iv in intervals:
        out.extend(_map_stitched_interval_to_orig(iv, chunks))
    return out

def whisper_transcribe_to_files(
    *,
    input_path: str,
    out_prefix: str,
    anchor_file: Optional[str] = None,
    reported_input: Optional[str] = None,
    model_name: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    task: str,
    beam_size: int,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    log_every_s: float,
    vad_filter: bool,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 1000,
    vad_speech_pad_ms: int = 400,
    chunk_length_s: int = 30,
    log_prob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    compression_ratio_threshold: Optional[float] = 2.4,
    hallucination_silence_threshold: Optional[float] = None,
    clip_timestamps: List[float] | str,
    vad_keep: Optional[List[Interval]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    section_minutes: int = 30,
    batched: bool = False,
    batch_size: int = 16,
) -> None:
    configure_hf_cache()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die("Missing dependency: faster-whisper. Install with: pip install faster-whisper")

    out_txt = f"{out_prefix}.txt"
    out_json = f"{out_prefix}.json"
    out_timestamped = f"{out_prefix}_timestamped.txt"
    out_sectioned = f"{out_prefix}_sectioned.txt"

    if anchor_file is None:
        anchor_file = input_path
    if reported_input is None:
        reported_input = input_path

    initial_prompt2 = None
    if initial_prompt is not None:
        initial_prompt2 = (initial_prompt or "").strip()
        if initial_prompt2 == "":
            initial_prompt2 = None

    hotwords2 = None
    if hotwords is not None:
        hotwords2 = (hotwords or "").strip()
        if hotwords2 == "":
            hotwords2 = None

    total_s = 0.0
    try:
        # Prefer the "real" recording's duration for progress/anchoring even if Whisper runs on a temp file.
        total_s = ffprobe_duration_seconds(anchor_file)
    except Exception:
        total_s = 0.0

    dev = _choose_whisper_device(device)
    if dev == "cpu":
        compute_type2 = _whisper_cpu_compute_type(compute_type)
        if compute_type2 != compute_type:
            print(f"[whisper] cpu compute_type override: {compute_type} -> {compute_type2}", file=sys.stderr)
            compute_type = compute_type2
    print(f"[whisper] model={model_name} device={dev} compute_type={compute_type}", file=sys.stderr)
    try:
        model = WhisperModel(model_name, device=dev, compute_type=compute_type)
    except Exception as e:
        if dev == "cuda":
            die(
                "CUDA init failed. Not falling back to CPU.\n"
                f"Error: {e}\n"
                "\n"
                "Common fixes:\n"
                "  - Ensure NVIDIA driver is installed and working (nvidia-smi should succeed).\n"
                "  - Ensure cuBLAS is available. In this repo's venv you can install it with:\n"
                "      ./.venv/bin/python -m pip install -U nvidia-cublas-cu12\n"
                "    The ./diarise wrapper will automatically add nvidia/*/lib to LD_LIBRARY_PATH.\n"
                "  - If you want CPU explicitly, re-run with: --whisper-device cpu\n"
            )
        raise

    # faster-whisper built-in Silero VAD defaults to threshold=0.5 and (in batched mode) min_silence_duration_ms=160.
    # We default to threshold=0.5 and min_silence_duration_ms=1000, configurable via CLI flags.
    vad_parameters = None
    if vad_filter and (not isinstance(clip_timestamps, list)):
        vad_parameters = {
            "threshold": float(vad_threshold),
            "min_silence_duration_ms": int(vad_min_silence_ms),
            "speech_pad_ms": int(vad_speech_pad_ms),
        }

    max_chunk_len = int(chunk_length_s)
    if max_chunk_len <= 0 or max_chunk_len > 30:
        die("--whisper-chunk-length must be between 1 and 30 seconds")

    def _is_cuda_oom(e: BaseException) -> bool:
        s = str(e).lower()
        return ("out of memory" in s) or ("cublas_status_alloc_failed" in s)

    def _is_cuda_runtime_issue(e: BaseException) -> bool:
        # Covers common late-fail cases where CUDA loads lazily (e.g. missing libcublas at first GEMM).
        s = str(e).lower()
        return (
            ("libcublas.so" in s)
            or ("cublas" in s)
            or ("cudart" in s)
            or ("cuda failed" in s)
            or ("cuda runtime" in s)
        )

    def do_transcribe() -> Tuple[Iterable[Any], Any]:
        if batched:
            from faster_whisper.transcribe import BatchedInferencePipeline

            pipe = BatchedInferencePipeline(model)
            clip_dicts = None
            vf = vad_filter
            if isinstance(clip_timestamps, list):
                # Batched pipeline expects clip_timestamps as [{"start":s,"end":e}, ...] in seconds.
                ivs = [(float(clip_timestamps[i]), float(clip_timestamps[i + 1])) for i in range(0, len(clip_timestamps), 2)]
                # Batched pipeline will truncate segments > chunk_length. Split to be safe.
                ivs = _split_intervals_max_len(ivs, float(max_chunk_len))
                clip_dicts = _clip_timestamps_dicts(ivs)
                vf = False  # ignored when clip_timestamps is set, but keep explicit.
            return pipe.transcribe(
                input_path,
                language=language,
                task=task,
                beam_size=beam_size,
                initial_prompt=initial_prompt2,
                hotwords=hotwords2,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                vad_filter=vf,
                vad_parameters=vad_parameters if vf else None,
                clip_timestamps=clip_dicts,
                without_timestamps=False,
                batch_size=int(batch_size),
                chunk_length=max_chunk_len,
                hallucination_silence_threshold=hallucination_silence_threshold,
            )

        ct2: List[float] | str = clip_timestamps
        if isinstance(clip_timestamps, list):
            ivs = [(float(clip_timestamps[i]), float(clip_timestamps[i + 1])) for i in range(0, len(clip_timestamps), 2)]
            ivs = _split_intervals_max_len(ivs, float(max_chunk_len))
            ct2 = _intervals_to_clip_timestamps(ivs)

        return model.transcribe(
            input_path,
            language=language,
            task=task,
            beam_size=beam_size,
            initial_prompt=initial_prompt2,
            hotwords=hotwords2,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None,
            clip_timestamps=ct2,
            chunk_length=max_chunk_len,
            hallucination_silence_threshold=hallucination_silence_threshold,
        )

    t0 = time.time()
    segments = None
    info = None
    while True:
        try:
            segments, info = do_transcribe()
            break
        except RuntimeError as e:
            # Some CUDA setups can initialize but fail when the first GEMM runs (e.g. cuBLAS errors).
            if dev == "cuda":
                if batched and _is_cuda_oom(e):
                    die(
                        "CUDA out of memory in --batched mode. Re-run with a smaller --batch-size "
                        "(e.g. 8 for medium) and/or smaller --whisper-beam-size."
                    )
                die(
                    "CUDA runtime failed during transcription. Not falling back to CPU.\n"
                    f"Error: {e}\n"
                    "\n"
                    "If this mentions libcublas.so.12, install cuBLAS into the venv:\n"
                    "  ./.venv/bin/python -m pip install -U nvidia-cublas-cu12\n"
                    "\n"
                    "If this is an out-of-memory error, lower one or more of:\n"
                    "  - --batch-size (for --batched)\n"
                    "  - --whisper-beam-size\n"
                    "  - --whisper-model (e.g. base/small instead of medium)\n"
                    "\n"
                    "If you want CPU explicitly, re-run with: --whisper-device cpu\n"
                )

            # BatchedInferencePipeline requires either vad_filter=True or clip_timestamps for long audio.
            # If the user disabled VAD in batched mode, retry using the non-batched pipeline.
            if batched and "No clip timestamps found" in str(e):
                print(f"[whisper] batched mode requires VAD; retrying non-batched (vad_filter={vad_filter})", file=sys.stderr)
                segments, info = model.transcribe(
                    input_path,
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    initial_prompt=initial_prompt2,
                    hotwords=hotwords2,
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters if vad_filter else None,
                    clip_timestamps=clip_timestamps,
                )
                break
            raise
        except AssertionError as e:
            # BatchedInferencePipeline can assert on some corrupt inputs (e.g. container duration > decodable audio).
            # Fall back to the non-batched pipeline so the run still completes.
            if batched:
                print(f"[whisper] batched pipeline failed ({e}); retrying non-batched", file=sys.stderr)
                segments, info = model.transcribe(
                    input_path,
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    initial_prompt=initial_prompt2,
                    hotwords=hotwords2,
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters if vad_filter else None,
                    clip_timestamps=clip_timestamps,
                )
                break
            raise

    segs_out: List[Dict[str, Any]] = []
    texts: List[str] = []
    next_log_at = log_every_s if log_every_s > 0 else float("inf")
    try:
        assert segments is not None
        for s in segments:
            end_t = float(s.end)
            if end_t >= next_log_at:
                if total_s > 0:
                    pct = max(0.0, min(100.0, (end_t / total_s) * 100.0))
                    print(f"[whisper] {end_t/3600:.2f}h / {total_s/3600:.2f}h ({pct:.1f}%)", file=sys.stderr, flush=True)
                else:
                    print(f"[whisper] {end_t/3600:.2f}h", file=sys.stderr, flush=True)
                next_log_at += log_every_s

            segs_out.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            txt = (s.text or "").strip()
            if txt:
                texts.append(txt)
    except RuntimeError as e:
        # BatchedInferencePipeline can fail lazily while iterating the generator. If CUDA dies mid-run,
        # retry from scratch on CPU rather than dumping a stack trace.
        if dev == "cuda":
            if batched and _is_cuda_oom(e):
                die(
                    "CUDA out of memory during batched transcription. Re-run with a smaller --batch-size "
                    "(e.g. 8 for medium) and/or smaller --whisper-beam-size."
                )
            if _is_cuda_runtime_issue(e):
                print(f"[whisper] CUDA runtime failed while iterating segments ({e}); retrying on CPU", file=sys.stderr)
                dev = "cpu"
                compute_type = _whisper_cpu_compute_type(compute_type)
                model = WhisperModel(model_name, device=dev, compute_type=compute_type)
                # Restart from scratch.
                segs_out = []
                texts = []
                next_log_at = log_every_s if log_every_s > 0 else float('inf')
                segments, info = do_transcribe()
                for s in segments:
                    end_t = float(s.end)
                    if end_t >= next_log_at:
                        if total_s > 0:
                            pct = max(0.0, min(100.0, (end_t / total_s) * 100.0))
                            print(f"[whisper] {end_t/3600:.2f}h / {total_s/3600:.2f}h ({pct:.1f}%)", file=sys.stderr, flush=True)
                        else:
                            print(f"[whisper] {end_t/3600:.2f}h", file=sys.stderr, flush=True)
                        next_log_at += log_every_s
                    segs_out.append({"start": float(s.start), "end": float(s.end), "text": s.text})
                    txt = (s.text or "").strip()
                    if txt:
                        texts.append(txt)
                # Continue writing outputs.
            else:
                raise
        else:
            raise

    ensure_parent_dir(out_txt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(texts) + "\n")

    ensure_parent_dir(out_json)
    anchor_dt, anchor_src = get_file_anchor_time(anchor_file)
    anchor_tail_ok, anchor_tail_note = ffmpeg_tail_decodes_ok(anchor_file)
    anchor_duration_s = ffprobe_duration_seconds(anchor_file)
    payload: Dict[str, Any] = {
        "input": os.path.abspath(reported_input),
        "whisper_input": os.path.abspath(input_path),
        "anchor_file": os.path.abspath(anchor_file),
        "anchor_time": anchor_dt.isoformat(timespec="seconds"),
        "anchor_time_src": anchor_src,
        "anchor_time_assumed": "end_of_recording",
        "anchor_time_uncertain": (not anchor_tail_ok),
        "anchor_time_uncertain_note": (None if anchor_tail_ok else anchor_tail_note),
        "anchor_duration_s": float(anchor_duration_s),
        "model": model_name,
        "initial_prompt": initial_prompt2,
        "hotwords": hotwords2,
        "device": dev,
        "compute_type": compute_type,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "batched": bool(batched),
        "batch_size": int(batch_size) if batched else None,
        "vad_filter": bool(vad_filter),
        "clip_timestamps": clip_timestamps if isinstance(clip_timestamps, list) else None,
        "vad_keep": vad_keep if vad_keep is not None else None,
        "segments": segs_out,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if extra_payload:
        payload.update(extra_payload)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if section_minutes > 0:
        try:
            write_sectioned_transcript(
                out_path=out_sectioned,
                anchor_file=anchor_file,
                segments=segs_out,
                bin_minutes=section_minutes,
                tail_probe=(anchor_tail_ok, anchor_tail_note),
            )
            print(f"[whisper] wrote {out_sectioned}", file=sys.stderr)
            write_timestamped_transcript(
                out_path=out_timestamped,
                anchor_file=anchor_file,
                segments=segs_out,
                bin_minutes=section_minutes,
                tail_probe=(anchor_tail_ok, anchor_tail_note),
            )
            print(f"[whisper] wrote {out_timestamped}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] failed to write sectioned/timestamped transcript: {e}", file=sys.stderr)

    dt = time.time() - t0
    if section_minutes > 0:
        print(f"[done] wrote {out_txt}, {out_json}, {out_sectioned}, {out_timestamped} in {dt/60:.1f} min", file=sys.stderr)
    else:
        print(f"[done] wrote {out_txt} and {out_json} in {dt/60:.1f} min", file=sys.stderr)


# -----------------------------
# Interval logic
# -----------------------------

Interval = Tuple[float, float]  # (start_sec, end_sec), end > start

def merge_intervals(intervals: List[Interval], gap: float = 0.0) -> List[Interval]:
    """Merge overlapping intervals, and also merge if separated by <= gap."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    out: List[Interval] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out

def clamp_intervals(intervals: List[Interval], lo: float, hi: float) -> List[Interval]:
    out: List[Interval] = []
    for s, e in intervals:
        s2 = max(lo, s)
        e2 = min(hi, e)
        if e2 > s2:
            out.append((s2, e2))
    return out

def complement_intervals(removed: List[Interval], total_dur: float) -> List[Interval]:
    """Return keep intervals for [0,total_dur] minus removed."""
    removed = merge_intervals(clamp_intervals(removed, 0.0, total_dur), gap=0.0)
    keep: List[Interval] = []
    cur = 0.0
    for s, e in removed:
        if s > cur:
            keep.append((cur, s))
        cur = max(cur, e)
    if cur < total_dur:
        keep.append((cur, total_dur))
    return keep

def drop_short(intervals: List[Interval], min_len: float) -> List[Interval]:
    return [(s, e) for s, e in intervals if (e - s) >= min_len]


# -----------------------------
# Prepass: stream PCM and mark dead-air runs
# -----------------------------

@dataclass
class PrepassConfig:
    sr: int = 16000
    window_s: float = 0.50        # analysis window (seconds)
    min_dead_s: float = 10.0      # mark dead-air runs >= this
    rms_db: float = -60.0         # dead-air if RMS below this
    peak_db: float = -55.0        # and Peak below this
    highpass_hz: float = 80.0     # reduce rumble biasing the RMS/peak
    edge_keep_s: float = 0.40     # keep this much on each side of removed dead-air run
    progress_every_s: float = 300.0  # log every N seconds of audio processed (0 disables)

def _dbfs_from_lin(x: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(x, eps))

def detect_dead_air_intervals(input_path: str, dur_s: float, cfg: PrepassConfig) -> List[Interval]:
    """
    Stream-decode to s16le mono cfg.sr, compute windowed peak/RMS, and detect long runs
    where both are below thresholds.
    """
    bytes_per_sample = 2  # s16le
    win_samples = max(1, int(cfg.window_s * cfg.sr))
    win_bytes = win_samples * bytes_per_sample

    ff_cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-af", f"highpass=f={cfg.highpass_hz:g}",
        "-ac", "1", "-ar", str(cfg.sr),
        "-f", "s16le", "-"
    ]

    # Don't pipe ffmpeg stderr without draining it: long/continuous decode errors can fill the pipe
    # and deadlock the whole streaming loop. Write stderr to a temp file instead.
    ffmpeg_stderr = tempfile.NamedTemporaryFile(prefix="ffmpeg_pcm_", suffix=".log", delete=False)
    proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ffmpeg_stderr)
    if proc.stdout is None:
        die("Failed to start ffmpeg for PCM streaming")

    removed: List[Interval] = []
    silent_run_start: Optional[float] = None

    processed_samples = 0
    next_log_at = cfg.progress_every_s if cfg.progress_every_s > 0 else float("inf")

    def read_exact(n: int) -> bytes:
        # Read up to n bytes; may return fewer at EOF.
        return proc.stdout.read(n)

    while True:
        buf = read_exact(win_bytes)
        if not buf:
            break

        # Allow partial final window
        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            break

        # Normalize to [-1, 1)
        x = samples / 32768.0
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x * x)))

        peak_db = _dbfs_from_lin(peak)
        rms_db = _dbfs_from_lin(rms)

        is_dead = (peak_db < cfg.peak_db) and (rms_db < cfg.rms_db)

        t0 = processed_samples / cfg.sr
        processed_samples += samples.size
        t1 = processed_samples / cfg.sr

        if cfg.progress_every_s > 0 and t1 >= next_log_at:
            print(f"[prepass] scanned {t1/3600:.2f}h / {dur_s/3600:.2f}h", file=sys.stderr)
            next_log_at += cfg.progress_every_s

        if is_dead:
            if silent_run_start is None:
                silent_run_start = t0
        else:
            if silent_run_start is not None:
                run_end = t0
                if (run_end - silent_run_start) >= cfg.min_dead_s:
                    # Remove only the "interior" of this dead-air run to protect boundaries.
                    s = silent_run_start + cfg.edge_keep_s
                    e = run_end - cfg.edge_keep_s
                    if e > s:
                        removed.append((s, e))
                silent_run_start = None

    # Finalize if file ends during dead run
    if silent_run_start is not None:
        run_end = processed_samples / cfg.sr
        if (run_end - silent_run_start) >= cfg.min_dead_s:
            s = silent_run_start + cfg.edge_keep_s
            e = run_end - cfg.edge_keep_s
            if e > s:
                removed.append((s, e))

    # Clean up ffmpeg process
    proc.stdout.close()
    proc.wait()
    ffmpeg_stderr.close()
    try:
        if proc.returncode != 0:
            try:
                with open(ffmpeg_stderr.name, "r", encoding="utf-8", errors="ignore") as f:
                    stderr = f.read().strip()
            except OSError:
                stderr = ""
            die(f"ffmpeg PCM streaming failed:\n{stderr}")
    finally:
        try:
            os.unlink(ffmpeg_stderr.name)
        except OSError:
            pass

    removed = merge_intervals(removed, gap=0.0)
    removed = clamp_intervals(removed, 0.0, dur_s)
    return removed


# -----------------------------
# Rendering: build ffmpeg filter script (atrim + concat)
# -----------------------------

def write_concat_filter_script(keep: List[Interval], script_path: str, fade_s: float) -> None:
    """
    Generates an ffmpeg -filter_complex_script that:
      - atrim each keep interval
      - applies short fades to avoid clicks
      - concat them
    """
    keep = drop_short(keep, min_len=0.010)
    if not keep:
        die("No keep intervals left to render (audio would be empty).")

    lines: List[str] = []
    for i, (s, e) in enumerate(keep):
        dur = e - s
        # If segment is very short, skip fades to avoid negative times.
        fd = 0.0
        if fade_s > 0 and dur > 2.5 * fade_s:
            fd = fade_s

        chain = f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=N/SR/TB"
        if fd > 0.0:
            # Fade-in always starts at 0
            chain += f",afade=t=in:st=0:d={fd:.6f}"
            # Fade-out starts at dur - fd
            chain += f",afade=t=out:st={(dur - fd):.6f}:d={fd:.6f}"
        chain += f"[a{i}];"
        lines.append(chain)

    concat_in = "".join([f"[a{i}]" for i in range(len(keep))])
    lines.append(f"{concat_in}concat=n={len(keep)}:v=0:a=1[outa]")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def ffmpeg_render_keep_intervals(
    input_path: str,
    keep: List[Interval],
    output_path: str,
    *,
    fade_s: float,
    audio_codec: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    """
    Renders keep intervals from input into output using a generated filter script.
    """
    ensure_parent_dir(output_path)

    with tempfile.TemporaryDirectory(prefix="condense_audio_") as td:
        script_path = os.path.join(td, "filter.txt")
        write_concat_filter_script(keep, script_path, fade_s=fade_s)

        # Use -progress so long renders show activity and don't look hung.
        total_out_s = sum((e - s) for s, e in keep)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-v", "error",
            "-progress", "pipe:2",
            "-i", input_path,
            "-filter_complex_script", script_path,
            "-map", "[outa]"
        ]
        if audio_codec:
            cmd += ["-c:a", audio_codec]
        if extra_args:
            cmd += extra_args
        cmd += [output_path]
        run_ffmpeg_with_progress(cmd, label="render", total_out_s=total_out_s)


# -----------------------------
# Silero VAD pass (streaming)
# -----------------------------

@dataclass
class VADConfig:
    sr: int = 16000
    threshold: float = 0.30
    block_samples: int = 512            # VADIterator works well with 512 @ 16k
    min_speech_s: float = 0.20
    pad_pre_s: float = 0.35
    pad_post_s: float = 0.60
    max_gap_to_keep_s: float = 10.0     # keep pauses up to this (avoid choppiness)
    device: str = "cpu"                 # "cpu" or "cuda"

@dataclass
class WebRTCVADConfig:
    aggressiveness: int = 2             # 0..3 (higher => more aggressive filtering)
    frame_ms: int = 30                  # 10/20/30 only
    min_silence_ms: int = 300           # close a speech segment after this much continuous non-speech

def silero_vad_speech_segments_streaming_ffmpeg(input_path: str, dur_s: float, cfg: VADConfig) -> List[Interval]:
    """
    Decode `input_path` -> 16kHz mono s16le with ffmpeg and run Silero VAD over the PCM stream.
    Returns keep intervals in seconds.
    """
    try:
        import torch
    except ImportError:
        die("Missing Python package: torch")

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )

    VADIterator = None
    if isinstance(utils, (list, tuple)):
        for u in utils:
            if getattr(u, "__name__", "") == "VADIterator":
                VADIterator = u
                break
        if VADIterator is None and len(utils) >= 4 and getattr(utils[3], "__name__", "") == "VADIterator":
            VADIterator = utils[3]
    if VADIterator is None:
        die("Could not find VADIterator in Silero utils (repo changed?)")

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    model.eval()

    # Stream-decode to PCM. Don't pipe stderr without draining it.
    ffmpeg_stderr = tempfile.NamedTemporaryFile(prefix="ffmpeg_vad_", suffix=".log", delete=False)
    ff_cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-ac", "1", "-ar", str(cfg.sr),
        "-f", "s16le", "-"
    ]
    proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ffmpeg_stderr)
    if proc.stdout is None:
        die("Failed to start ffmpeg for VAD PCM streaming")

    samples_per_block = max(1, int(cfg.block_samples))
    bytes_per_block = samples_per_block * 2  # s16le

    vad_it = VADIterator(model, threshold=cfg.threshold, sampling_rate=cfg.sr)
    if hasattr(model, "reset_states"):
        model.reset_states()

    segments: List[Interval] = []
    current_start_s: Optional[float] = None

    processed_samples = 0
    next_log_at = 300.0

    with torch.no_grad():
        while True:
            buf = proc.stdout.read(bytes_per_block)
            if not buf:
                break
            if len(buf) < bytes_per_block:
                # Pad final short chunk so the model doesn't error.
                buf = buf + b"\x00" * (bytes_per_block - len(buf))

            x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            processed_samples += x.size

            t_s = processed_samples / cfg.sr
            if t_s >= next_log_at:
                total = dur_s if dur_s > 0 else t_s
                print(f"[vad] processed {t_s/3600:.2f}h / {total/3600:.2f}h", file=sys.stderr)
                next_log_at += 300.0

            audio = torch.from_numpy(x).to(device)
            ev = vad_it(audio)  # returns sample indices by default
            if not ev:
                continue
            if isinstance(ev, dict):
                if "start" in ev:
                    current_start_s = float(ev["start"]) / float(cfg.sr)
                if "end" in ev and current_start_s is not None:
                    segments.append((current_start_s, float(ev["end"]) / float(cfg.sr)))
                    current_start_s = None

    total_dur = min(float(dur_s), processed_samples / cfg.sr) if dur_s else (processed_samples / cfg.sr)
    if current_start_s is not None:
        segments.append((current_start_s, total_dur))

    proc.stdout.close()
    proc.wait()
    ffmpeg_stderr.close()
    try:
        if proc.returncode != 0:
            try:
                with open(ffmpeg_stderr.name, "r", encoding="utf-8", errors="ignore") as f:
                    stderr = f.read().strip()
            except OSError:
                stderr = ""
            die(f"ffmpeg VAD PCM streaming failed:\n{stderr}")
    finally:
        try:
            os.unlink(ffmpeg_stderr.name)
        except OSError:
            pass

    segments = clamp_intervals(segments, 0.0, total_dur)
    segments = drop_short(segments, cfg.min_speech_s)

    padded: List[Interval] = []
    for s, e in segments:
        padded.append((s - cfg.pad_pre_s, e + cfg.pad_post_s))
    padded = clamp_intervals(padded, 0.0, total_dur)
    merged = merge_intervals(padded, gap=cfg.max_gap_to_keep_s)
    merged = drop_short(merged, cfg.min_speech_s)
    return merged

def _write_ffmpeg_concat_list(path: str, keep_intervals: List[Interval], out_path: str) -> None:
    """
    Writes an ffmpeg concat demuxer list that plays only [start,end] from `path` for each keep interval.
    """
    abs_path = os.path.abspath(path)
    lines: List[str] = []
    for s, e in keep_intervals:
        s2 = float(s)
        e2 = float(e)
        if e2 <= s2:
            continue
        # Avoid quoting complexity; repo paths have no spaces. If your path has spaces, move the file or
        # use a symlink without spaces.
        lines.append(f"file {abs_path}")
        lines.append(f"inpoint {s2:.6f}")
        lines.append(f"outpoint {e2:.6f}")
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

def webrtc_vad_speech_segments_streaming_ffmpeg(
    input_path: str,
    dur_s: float,
    cfg: VADConfig,
    wcfg: WebRTCVADConfig,
    *,
    concat_list: Optional[str] = None,
) -> List[Interval]:
    """
    Decode `input_path` -> 16kHz mono s16le with ffmpeg and run WebRTC VAD over the PCM stream.
    Returns keep intervals in seconds.
    """
    try:
        import webrtcvad
    except ImportError:
        die("Missing Python package: webrtcvad (pip install webrtcvad)")

    if wcfg.frame_ms not in (10, 20, 30):
        die("WebRTC VAD frame size must be 10, 20, or 30 ms")
    if not (0 <= wcfg.aggressiveness <= 3):
        die("WebRTC VAD aggressiveness must be 0..3")

    vad = webrtcvad.Vad(wcfg.aggressiveness)
    frame_s = wcfg.frame_ms / 1000.0
    samples_per_frame = int(cfg.sr * frame_s)
    bytes_per_frame = samples_per_frame * 2  # s16le mono
    min_silence_frames = max(1, int(round((wcfg.min_silence_ms / 1000.0) / frame_s)))

    ffmpeg_stderr = tempfile.NamedTemporaryFile(prefix="ffmpeg_webrtc_", suffix=".log", delete=False)
    if concat_list:
        ff_cmd = [
            "ffmpeg", "-v", "error",
            "-safe", "0",
            "-f", "concat",
            "-i", concat_list,
            "-vn",
            "-ac", "1", "-ar", str(cfg.sr),
            "-f", "s16le", "-"
        ]
    else:
        ff_cmd = [
            "ffmpeg", "-v", "error",
            "-i", input_path,
            "-vn",
            "-ac", "1", "-ar", str(cfg.sr),
            "-f", "s16le", "-"
        ]
    proc = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ffmpeg_stderr)
    if proc.stdout is None:
        die("Failed to start ffmpeg for WebRTC PCM streaming")

    segments: List[Interval] = []
    in_speech = False
    speech_start = 0.0
    silence_run = 0
    processed_frames = 0
    next_log_at = 300.0

    while True:
        buf = proc.stdout.read(bytes_per_frame)
        if not buf:
            break
        if len(buf) < bytes_per_frame:
            buf = buf + b"\x00" * (bytes_per_frame - len(buf))

        t_end = (processed_frames + 1) * frame_s
        if t_end >= next_log_at:
            total = dur_s if dur_s > 0 else t_end
            print(f"[vad] processed {t_end/3600:.2f}h / {total/3600:.2f}h", file=sys.stderr)
            next_log_at += 300.0

        is_speech = vad.is_speech(buf, cfg.sr)
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
    proc.wait()
    ffmpeg_stderr.close()
    try:
        if proc.returncode != 0:
            try:
                with open(ffmpeg_stderr.name, "r", encoding="utf-8", errors="ignore") as f:
                    stderr = f.read().strip()
            except OSError:
                stderr = ""
            die(f"ffmpeg WebRTC PCM streaming failed:\n{stderr}")
    finally:
        try:
            os.unlink(ffmpeg_stderr.name)
        except OSError:
            pass

    segments = clamp_intervals(segments, 0.0, total_dur)
    segments = drop_short(segments, cfg.min_speech_s)

    padded: List[Interval] = []
    for s, e in segments:
        padded.append((s - cfg.pad_pre_s, e + cfg.pad_post_s))
    padded = clamp_intervals(padded, 0.0, total_dur)
    merged = merge_intervals(padded, gap=cfg.max_gap_to_keep_s)
    merged = drop_short(merged, cfg.min_speech_s)
    return merged

def silero_vad_speech_segments_streaming(wav_16k_mono_s16: str, cfg: VADConfig) -> List[Interval]:
    """
    Streaming Silero VAD using torch.hub + VADIterator to avoid loading hours into memory.
    """
    try:
        import torch
    except ImportError:
        die("Missing Python package: torch")

    # Load Silero VAD model + utilities
    # Cached after first run.
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )

    # Expect utils to include VADIterator
    if not isinstance(utils, (list, tuple)) or len(utils) < 5:
        die("Unexpected Silero utils layout from torch.hub (repo changed?)")

    # Older/typical layout:
    # (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    VADIterator = None
    for u in utils:
        if getattr(u, "__name__", "") == "VADIterator":
            VADIterator = u
            break
    if VADIterator is None:
        # Commonly it’s at index 3
        if len(utils) >= 4 and getattr(utils[3], "__name__", "") == "VADIterator":
            VADIterator = utils[3]
        else:
            die("Could not find VADIterator in Silero utils (repo changed?)")

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    model.eval()

    # Open PCM wav and stream frames
    with wave.open(wav_16k_mono_s16, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != cfg.sr or wf.getsampwidth() != 2:
            die("VAD input must be 16kHz mono s16 WAV (internal bug: wrong preprocessing).")

        total_frames = wf.getnframes()
        total_dur = total_frames / cfg.sr
        next_log_at = 300.0

        # Create iterator; signature has changed across revisions, so keep it simple:
        # threshold is usually passed into VADIterator(...) or via attribute.
        try:
            vad_it = VADIterator(model, threshold=cfg.threshold, sampling_rate=cfg.sr)
        except TypeError:
            try:
                vad_it = VADIterator(model, threshold=cfg.threshold)
            except TypeError:
                vad_it = VADIterator(model)

        # Ensure model state is clean
        if hasattr(model, "reset_states"):
            model.reset_states()

        segments: List[Interval] = []
        current_start: Optional[float] = None

        # Read in blocks
        frames_per_block = cfg.block_samples
        with torch.no_grad():
            while True:
                b = wf.readframes(frames_per_block)
                if not b:
                    break
                # Lightweight progress log (every ~5 minutes of audio).
                t = wf.tell() / cfg.sr
                if t >= next_log_at:
                    print(f"[vad] processed {t/3600:.2f}h / {total_dur/3600:.2f}h", file=sys.stderr)
                    next_log_at += 300.0
                x = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
                if x.size == 0:
                    break
                # Silero VAD expects chunks >= 512 samples at 16kHz. At EOF, `wave` can return
                # a short final chunk; zero-pad it so the model doesn't error.
                if x.size < cfg.block_samples:
                    x = np.pad(x, (0, cfg.block_samples - x.size), mode="constant")
                audio = torch.from_numpy(x).to(device)

                # VADIterator returns dicts with 'start'/'end' events. Newer versions support
                # return_seconds=True; older ones return sample indices. Prefer seconds, and
                # normalize to seconds in both cases.
                scale = 1.0
                try:
                    ev = vad_it(audio, return_seconds=True)
                except TypeError:
                    ev = vad_it(audio)
                    scale = 1.0 / float(cfg.sr)

                # Normalize possible return shapes.
                events = []
                if ev is None:
                    events = []
                elif isinstance(ev, dict):
                    events = [ev]
                elif isinstance(ev, (list, tuple)):
                    events = list(ev)
                else:
                    events = []

                for d in events:
                    if not isinstance(d, dict):
                        continue
                    if "start" in d:
                        current_start = float(d["start"]) * scale
                    if "end" in d and current_start is not None:
                        segments.append((current_start, float(d["end"]) * scale))
                        current_start = None

        # If we ended while "in speech", close it at end of file.
        if current_start is not None:
            segments.append((current_start, total_dur))

    # Postprocess: clamp, drop shorts, pad, and merge while keeping pauses <= max_gap_to_keep_s
    segments = clamp_intervals(segments, 0.0, float(total_dur))
    segments = drop_short(segments, cfg.min_speech_s)

    padded = []
    for s, e in segments:
        padded.append((s - cfg.pad_pre_s, e + cfg.pad_post_s))
    padded = clamp_intervals(padded, 0.0, float(total_dur))

    # Merge speech "islands" while preserving natural pauses up to max_gap_to_keep_s
    merged = merge_intervals(padded, gap=cfg.max_gap_to_keep_s)
    merged = drop_short(merged, cfg.min_speech_s)
    return merged


# -----------------------------
# Codec choice for final output
# -----------------------------

def guess_codec_from_ext(path: str) -> Tuple[Optional[str], List[str]]:
    """
    Returns (codec, extra_args). If codec is None, ffmpeg will choose default.
    """
    ext = os.path.splitext(path.lower())[1]
    if ext in (".wav",):
        return ("pcm_s16le", [])
    if ext in (".flac",):
        return ("flac", [])
    if ext in (".mp3",):
        return ("libmp3lame", ["-q:a", "2"])
    if ext in (".m4a", ".mp4"):
        return ("aac", ["-b:a", "192k"])
    if ext in (".opus",):
        return ("libopus", ["-b:a", "96k"])
    # unknown/let ffmpeg decide
    return (None, [])


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    which_or_die("ffmpeg")
    which_or_die("ffprobe")

    ap = argparse.ArgumentParser(description="Condense a long recording or output speech timestamps (VAD).")
    ap.add_argument("input", help="Input audio/video file, or a directory of files in transcription mode")
    ap.add_argument("output", nargs="?", default=None,
                    help="Final condensed output file (codec inferred from extension). "
                         "If INPUT is a directory in transcription mode, this is the output directory. "
                         "Optional with --timestamps-only.")

    # Prepass knobs
    ap.add_argument("--gap-cut", type=float, default=10.0,
                    help="Only remove non-speech/dead-air gaps longer than this many seconds (default: 10)")
    ap.add_argument("--prepass-window", type=float, default=0.50, help="Prepass analysis window seconds (default: 0.50)")
    ap.add_argument("--prepass-rms-db", type=float, default=-60.0, help="Prepass dead-air RMS threshold dBFS (default: -60)")
    ap.add_argument("--prepass-peak-db", type=float, default=-55.0, help="Prepass dead-air peak threshold dBFS (default: -55)")
    ap.add_argument("--prepass-highpass", type=float, default=80.0, help="Prepass highpass Hz (default: 80)")
    ap.add_argument("--prepass-edge-keep", type=float, default=0.40, help="Keep this much at edges of removed runs (default: 0.40s)")

    # VAD knobs
    ap.add_argument("--vad-threshold", type=float, default=0.30, help="Silero VAD threshold (lower => higher recall) (default: 0.30)")
    ap.add_argument("--vad-min-speech", type=float, default=0.20, help="Drop speech segments shorter than this (default: 0.20s)")
    ap.add_argument("--pad-pre", type=float, default=0.35, help="Pad before each VAD segment (default: 0.35s)")
    ap.add_argument("--pad-post", type=float, default=0.60, help="Pad after each VAD segment (default: 0.60s)")
    ap.add_argument("--vad-block", type=int, default=512, help="VAD block size in samples @16k (default: 512)")
    ap.add_argument("--vad-backend", default="silero", choices=["silero", "webrtc"],
                    help="VAD backend to use (default: silero)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for VAD model (default: cpu)")

    # WebRTC VAD knobs (used only with --vad-backend webrtc)
    ap.add_argument("--webrtc-aggressiveness", type=int, default=2, help="WebRTC VAD aggressiveness 0..3 (default: 2)")
    ap.add_argument("--webrtc-frame-ms", type=int, default=30, choices=[10, 20, 30], help="WebRTC VAD frame size ms (default: 30)")
    ap.add_argument("--webrtc-min-silence-ms", type=int, default=300,
                    help="Close a speech segment after this much continuous non-speech (default: 300ms)")

    # Rendering / output
    ap.add_argument("--timestamps-only", action="store_true",
                    help="Only detect speech timestamps and write JSON (skips prepass and audio rendering).")
    ap.add_argument("--condense-from-json", default=None,
                    help="Read keep intervals from a JSON report (expects key 'vad_keep') and render OUTPUT from INPUT.")
    ap.add_argument("--fade", type=float, default=0.02, help="Fade duration at cut points (seconds) (default: 0.02)")
    ap.add_argument("--keep-temp", action="store_true", help="Keep intermediate files (in a temp dir) for inspection")
    ap.add_argument("--report-json", default=None, help="Write a JSON report (intervals, settings) to this path")

    # Whisper transcription (end-to-end in this script; default mode when OUTPUT is omitted)
    ap.add_argument("--out-prefix", default=None,
                    help="Output prefix for transcription files (default: <input_stem>_whisper_<model>).")
    ap.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size/name (default: base). Supports aliases: large -> large-v3, large-turbo -> large-v3-turbo",
    )
    ap.add_argument("--whisper-device", default="cuda", choices=["auto", "cpu", "cuda"],
                    help="Device for Whisper inference (default: cuda)")
    ap.add_argument("--whisper-compute-type", default="float16",
                    help="CTranslate2 compute type, e.g. int8, int8_float16, float16, float32 (default: float16)")
    ap.add_argument("--whisper-language", default="en", help="Language code, e.g. en (default: en)")
    ap.add_argument("--whisper-task", default="transcribe", choices=["transcribe", "translate"], help="Task (default: transcribe)")
    ap.add_argument("--whisper-beam-size", type=int, default=5, help="Beam size (default: 5)")
    ap.add_argument("--whisper-initial-prompt", default="Hello.",
                    help="Initial prompt to seed Whisper context. Default: 'Hello.'. Use empty string to disable.")
    ap.add_argument("--whisper-hotwords", default="",
                    help="Hotwords to bias decoding across the whole file (e.g. 'jerboa jerboas gerbil'). Default: disabled.")
    ap.set_defaults(whisper_vad_filter=True)
    ap.add_argument("--whisper-vad-filter", dest="whisper_vad_filter", action="store_true",
                    help="Enable faster-whisper internal VAD filter (default: enabled)")
    ap.add_argument("--no-whisper-vad-filter", dest="whisper_vad_filter", action="store_false",
                    help="Disable faster-whisper internal VAD filter")
    ap.add_argument("--whisper-vad-threshold", type=float, default=0.5,
                    help="faster-whisper (Silero) VAD threshold (higher => more strict) (default: 0.5)")
    ap.add_argument("--whisper-vad-min-silence-ms", type=int, default=1000,
                    help="Minimum silence duration to split speech segments in Whisper VAD (default: 1000ms)")
    ap.add_argument("--whisper-vad-speech-pad-ms", type=int, default=400,
                    help="Padding added around speech segments in Whisper VAD (default: 400ms). Increase to reduce boundary word drops.")
    ap.add_argument("--whisper-chunk-length", type=int, default=30,
                    help="Whisper decode chunk length in seconds (1..30) (default: 30)")
    ap.add_argument("--whisper-log-prob-threshold", type=float, default=-1.0,
                    help="Drop/avoid low-confidence segments below this avg log prob (default: -1.0). Higher (e.g. -0.7) is stricter.")
    ap.add_argument("--whisper-no-speech-threshold", type=float, default=0.6,
                    help="No-speech probability threshold (default: 0.6). Lower is more aggressive at skipping non-speech.")
    ap.add_argument("--whisper-compression-ratio-threshold", type=float, default=2.4,
                    help="Compression ratio threshold for decoding fallback/hallucination handling (default: 2.4). Lower is stricter.")
    ap.add_argument("--whisper-hallucination-silence-threshold", type=float, default=None,
                    help="If set, treat long silence as a boundary to reduce hallucinations (seconds). Default: unset.")
    ap.add_argument("--whisper-log-every", type=float, default=300.0,
                    help="Whisper progress log interval seconds (default: 300). Set 0 to disable.")
    ap.add_argument("--bandpass", action="store_true",
                    help="Before Whisper, apply a 100-7000 Hz bandpass (2-pole highpass+lowpass ~= 40 dB/decade) via ffmpeg.")
    ap.add_argument("--batched", action="store_true",
                    help="Use faster-whisper BatchedInferencePipeline (batch_size=16).")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="In --batched mode, batch size for faster-whisper (default: 16). Lower it if you hit CUDA OOM.")
    ap.add_argument("--section-minutes", type=int, default=30,
                    help="Write <out_prefix>_sectioned.txt and <out_prefix>_timestamped.txt with this bin size (default: 30). Set 0 to disable.")
    ap.add_argument("--prevad", action="store_true",
                    help="In transcription mode, run dead-air prepass + WebRTC VAD first and pass intervals to Whisper via clip_timestamps (disables whisper VAD).")
    ap.add_argument("--postprocess-json", default=None,
                    help="Skip transcription and (re)generate *_sectioned.txt and *_timestamped.txt from an existing diarise Whisper JSON output.")

    args = ap.parse_args()

    input_path = args.input
    output_path = args.output

    input_is_file = os.path.isfile(input_path)
    input_is_dir = os.path.isdir(input_path)
    if not input_is_file and not input_is_dir:
        die(f"Input path not found: {input_path}")

    if args.postprocess_json and input_is_dir:
        die("--postprocess-json requires INPUT to be a single audio file (used as anchor).")

    if args.postprocess_json:
        if int(args.section_minutes) <= 0:
            die("--postprocess-json requires --section-minutes > 0")
        try:
            with open(args.postprocess_json, "r", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as e:
            die(f"Failed to read JSON: {e}")
        segs = j.get("segments")
        if not isinstance(segs, list):
            die("Invalid JSON: expected a top-level key 'segments' (list)")

        out_prefix = os.path.splitext(os.path.abspath(args.postprocess_json))[0]
        out_sectioned = f"{out_prefix}_sectioned.txt"
        out_timestamped = f"{out_prefix}_timestamped.txt"
        tail_probe = ffmpeg_tail_decodes_ok(input_path)
        write_sectioned_transcript(
            out_path=out_sectioned,
            anchor_file=input_path,
            segments=segs,
            bin_minutes=int(args.section_minutes),
            tail_probe=tail_probe,
        )
        write_timestamped_transcript(
            out_path=out_timestamped,
            anchor_file=input_path,
            segments=segs,
            bin_minutes=int(args.section_minutes),
            tail_probe=tail_probe,
        )
        print(f"[done] wrote {out_sectioned} and {out_timestamped}", file=sys.stderr)
        return

    pre_cfg = PrepassConfig(
        sr=16000,
        window_s=args.prepass_window,
        min_dead_s=args.gap_cut,
        rms_db=args.prepass_rms_db,
        peak_db=args.prepass_peak_db,
        highpass_hz=args.prepass_highpass,
        edge_keep_s=args.prepass_edge_keep,
        progress_every_s=300.0,
    )

    vad_cfg = VADConfig(
        sr=16000,
        threshold=args.vad_threshold,
        block_samples=args.vad_block,
        min_speech_s=args.vad_min_speech,
        pad_pre_s=args.pad_pre,
        pad_post_s=args.pad_post,
        max_gap_to_keep_s=args.gap_cut,
        device=args.device,
    )

    webrtc_cfg = WebRTCVADConfig(
        aggressiveness=args.webrtc_aggressiveness,
        frame_ms=args.webrtc_frame_ms,
        min_silence_ms=args.webrtc_min_silence_ms,
    )

    def run_transcription_for_input(single_input_path: str, out_prefix_override: Optional[str] = None) -> None:
        dur_s = ffprobe_duration_seconds(single_input_path)
        print(f"[info] input duration: {dur_s/3600:.2f} hours", file=sys.stderr)

        base = os.path.splitext(os.path.basename(single_input_path))[0]
        whisper_model = _resolve_whisper_model_name(args.whisper_model)
        out_prefix_local = out_prefix_override or args.out_prefix or f"{base}_whisper_{_safe_slug(whisper_model)}"

        vad_keep: Optional[List[Interval]] = None
        clip_timestamps: List[float] | str = "0"
        vad_filter = bool(args.whisper_vad_filter)
        extra_payload: Dict[str, Any] = {}

        # Optional bandpass pre-processing for Whisper only (does not affect VAD passes).
        whisper_input_path = single_input_path
        bandpass_ctx = None
        if args.bandpass:
            if args.keep_temp:
                bandpass_dir = tempfile.mkdtemp(prefix="transcribe_bandpass_keep_")
                print(f"[info] keeping bandpass intermediates in: {bandpass_dir}", file=sys.stderr)
            else:
                bandpass_ctx = tempfile.TemporaryDirectory(prefix="transcribe_bandpass_")
                bandpass_dir = bandpass_ctx.name  # type: ignore

            bandpass_out = os.path.join(bandpass_dir, "bandpass_100_7000_16k_mono.flac")
            print("[bandpass] rendering 100-7000 Hz bandpassed audio for Whisper", file=sys.stderr)
            ff_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-nostats",
                "-v", "error",
                "-progress", "pipe:2",
                # try to continue through minor container corruption where possible
                "-fflags", "+discardcorrupt",
                "-err_detect", "ignore_err",
                "-i", single_input_path,
                "-vn",
                # 2 poles ~= 12 dB/oct ~= 40 dB/decade rolloff
                "-af", "highpass=f=100:poles=2,lowpass=f=7000:poles=2",
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "flac",
                bandpass_out,
            ]
            run_ffmpeg_with_progress(ff_cmd, label="bandpass", total_out_s=dur_s, emit_every_s=15.0)
            whisper_input_path = bandpass_out
            extra_payload["bandpass"] = {
                "enabled": True,
                "source": "ffmpeg",
                "af": "highpass=f=100:poles=2,lowpass=f=7000:poles=2",
                "out": os.path.abspath(bandpass_out),
                "out_sr": 16000,
                "out_ch": 1,
            }

        if args.prevad:
            # Prepass dead-air removal (stitched) -> WebRTC VAD on prepassed audio -> map back to original timeline.
            if args.keep_temp:
                workdir = tempfile.mkdtemp(prefix="transcribe_prevad_keep_")
                print(f"[info] keeping prevad intermediates in: {workdir}", file=sys.stderr)
            else:
                workdir_ctx = tempfile.TemporaryDirectory(prefix="transcribe_prevad_")
                workdir = workdir_ctx.name  # type: ignore

            removed = detect_dead_air_intervals(single_input_path, dur_s, pre_cfg)
            keep_pre = complement_intervals(removed, dur_s)
            keep_pre = drop_short(keep_pre, 0.050)
            prepass_out_s = sum((e - s) for s, e in keep_pre)
            extra_payload["prepass_removed"] = removed
            extra_payload["prepass_keep"] = keep_pre

            # Fast path: don't render a stitched intermediate file.
            # Stream only the keep intervals via ffmpeg concat demuxer and run VAD on that PCM stream.
            concat_list = os.path.join(workdir, "prepass_keep.concat.txt")
            _write_ffmpeg_concat_list(single_input_path, keep_pre, concat_list)

            print("[prevad] running WebRTC VAD on prepass keep intervals (ffmpeg concat -> PCM stream)", file=sys.stderr)
            vad_keep_stitched = webrtc_vad_speech_segments_streaming_ffmpeg(
                single_input_path,
                prepass_out_s,
                vad_cfg,
                webrtc_cfg,
                concat_list=concat_list,
            )
            if not vad_keep_stitched:
                die("Pre-VAD produced no keep intervals; nothing to transcribe.")

            chunks, stitched_dur = _build_stitch_chunks(keep_pre)
            vad_keep_mapped = _map_stitched_intervals_to_orig(vad_keep_stitched, chunks)
            vad_keep_mapped = clamp_intervals(vad_keep_mapped, 0.0, dur_s)
            vad_keep_mapped = merge_intervals(vad_keep_mapped, gap=0.0)

            extra_payload["prevad"] = {
                "backend": "webrtc",
                "source": "concat_stream",
                "concat_list": os.path.abspath(concat_list),
                "stitched_duration_s": stitched_dur,
                "timeline_chunks": chunks,
                "vad_keep_stitched": vad_keep_stitched,
            }

            vad_keep = vad_keep_mapped
            clip_timestamps = _intervals_to_clip_timestamps(vad_keep)
            vad_filter = False

        whisper_transcribe_to_files(
            input_path=whisper_input_path,
            out_prefix=out_prefix_local,
            anchor_file=single_input_path,
            reported_input=single_input_path,
            model_name=whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            language=args.whisper_language,
            task=args.whisper_task,
            beam_size=args.whisper_beam_size,
            initial_prompt=args.whisper_initial_prompt,
            hotwords=args.whisper_hotwords,
            log_every_s=float(args.whisper_log_every),
            vad_filter=vad_filter,
            vad_threshold=float(args.whisper_vad_threshold),
            vad_min_silence_ms=int(args.whisper_vad_min_silence_ms),
            vad_speech_pad_ms=int(args.whisper_vad_speech_pad_ms),
            chunk_length_s=int(args.whisper_chunk_length),
            log_prob_threshold=float(args.whisper_log_prob_threshold),
            no_speech_threshold=float(args.whisper_no_speech_threshold),
            compression_ratio_threshold=float(args.whisper_compression_ratio_threshold),
            hallucination_silence_threshold=(
                None if args.whisper_hallucination_silence_threshold is None else float(args.whisper_hallucination_silence_threshold)
            ),
            clip_timestamps=clip_timestamps,
            vad_keep=vad_keep,
            extra_payload=extra_payload,
            section_minutes=int(args.section_minutes),
            batched=bool(args.batched),
            batch_size=int(args.batch_size),
        )

    # Directory batch transcription mode with resume/skip.
    if input_is_dir:
        if args.timestamps_only or args.condense_from_json or args.postprocess_json:
            die("Directory INPUT is supported only in transcription mode.")
        if args.out_prefix:
            die("Directory INPUT does not support --out-prefix. Use OUTPUT as the output directory.")
        if output_path is None:
            die("Directory INPUT requires OUTPUT to be an output directory.")

        out_dir = os.path.abspath(output_path)
        if os.path.exists(out_dir) and (not os.path.isdir(out_dir)):
            die(f"OUTPUT exists and is not a directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        audio_files = _list_audio_files_in_dir(input_path)
        if not audio_files:
            die(f"No supported audio/video files found in: {input_path}")

        model_slug = _safe_slug(_resolve_whisper_model_name(args.whisper_model))
        total = len(audio_files)
        done = 0
        skipped = 0
        failed = 0
        for idx, src in enumerate(audio_files, start=1):
            base = os.path.splitext(os.path.basename(src))[0]
            out_prefix = _batch_out_prefix_for_source(src, out_dir, model_slug)
            # Backward compatibility with earlier directory-mode naming.
            legacy_out_prefix = os.path.join(out_dir, f"{base}_whisper_{model_slug}")
            if _transcription_outputs_complete(out_prefix, int(args.section_minutes)):
                skipped += 1
                print(f"[batch {idx}/{total}] skip (already done): {src}", file=sys.stderr)
                continue
            if out_prefix != legacy_out_prefix and _transcription_outputs_complete(legacy_out_prefix, int(args.section_minutes)):
                skipped += 1
                print(f"[batch {idx}/{total}] skip (already done, legacy prefix): {src}", file=sys.stderr)
                continue

            print(f"[batch {idx}/{total}] transcribing: {src}", file=sys.stderr)
            try:
                run_transcription_for_input(src, out_prefix_override=out_prefix)
                done += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed += 1
                print(f"[batch {idx}/{total}] failed: {src}: {e}", file=sys.stderr)

        print(f"[batch] done={done} skipped={skipped} failed={failed}", file=sys.stderr)
        if failed > 0:
            die("Batch finished with failures. Fix errors and re-run; completed files will be skipped.")
        return

    # Default mode: if OUTPUT is omitted (and we're not in the explicit VAD/condense modes),
    # run Whisper transcription and write <out_prefix>.txt/.json.
    if (output_path is None) and (not args.timestamps_only) and (not args.condense_from_json):
        run_transcription_for_input(input_path)
        return

    if not input_is_file:
        die("This mode requires INPUT to be a single file.")

    dur_s = ffprobe_duration_seconds(input_path)
    print(f"[info] input duration: {dur_s/3600:.2f} hours", file=sys.stderr)

    report = {
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path) if output_path else None,
        "input_duration_s": dur_s,
        "settings": {
            "gap_cut_s": args.gap_cut,
            "prepass": pre_cfg.__dict__,
            "vad": vad_cfg.__dict__,
            "vad_backend": args.vad_backend,
            "webrtc": webrtc_cfg.__dict__,
            "fade_s": args.fade,
            "timestamps_only": bool(args.timestamps_only),
        },
        "prepass_removed": [],
        "prepass_keep": [],
        "vad_keep": [],
    }

    if args.condense_from_json:
        if args.timestamps_only:
            die("Cannot use --timestamps-only and --condense-from-json together.")
        if not output_path:
            die("Missing required output path (provide OUTPUT).")
        try:
            with open(args.condense_from_json, "r", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as e:
            die(f"Failed to read JSON: {e}")
        keep = j.get("vad_keep")
        if not isinstance(keep, list) or not keep:
            die("JSON does not contain a non-empty 'vad_keep' list.")
        try:
            keep_intervals: List[Interval] = [(float(s), float(e)) for s, e in keep]
        except Exception:
            die("Invalid interval format in 'vad_keep' (expected [[start,end], ...]).")

        # Preserve input sampling rate/channels explicitly so output matches original.
        in_sr, in_ch = ffprobe_audio_params(input_path)
        codec, extra = guess_codec_from_ext(output_path)
        extra2 = list(extra) + ["-ar", str(in_sr), "-ac", str(in_ch)]

        print(f"[render] rendering final output from JSON -> {output_path}", file=sys.stderr)
        ffmpeg_render_keep_intervals(
            input_path,
            keep_intervals,
            output_path,
            fade_s=args.fade,
            audio_codec=codec,
            extra_args=extra2,
        )

        if args.report_json:
            report["vad_keep"] = keep_intervals
            ensure_parent_dir(args.report_json)
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[info] wrote report: {args.report_json}", file=sys.stderr)

        print("[done]", file=sys.stderr)
        return

    if args.timestamps_only:
        report_path = args.report_json
        if not report_path:
            base = os.path.splitext(os.path.basename(input_path))[0]
            report_path = f"{base}_vad.json"

        if args.vad_backend == "silero":
            print("[vad] running Silero VAD (ffmpeg PCM stream)", file=sys.stderr)
            vad_keep = silero_vad_speech_segments_streaming_ffmpeg(input_path, dur_s, vad_cfg)
        else:
            print("[vad] running WebRTC VAD (ffmpeg PCM stream)", file=sys.stderr)
            vad_keep = webrtc_vad_speech_segments_streaming_ffmpeg(input_path, dur_s, vad_cfg, webrtc_cfg)
        report["vad_keep"] = vad_keep

        ensure_parent_dir(report_path)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[info] wrote report: {report_path}", file=sys.stderr)
        print("[done]", file=sys.stderr)
        return

    if not output_path:
        die("Missing required output path (provide OUTPUT or use --timestamps-only).")

    # Work dir
    if args.keep_temp:
        workdir = tempfile.mkdtemp(prefix="condense_audio_keep_")
        print(f"[info] keeping intermediates in: {workdir}", file=sys.stderr)
    else:
        workdir_ctx = tempfile.TemporaryDirectory(prefix="condense_audio_")
        workdir = workdir_ctx.name  # type: ignore

    try:
        # 1) Prepass detect + render
        removed = detect_dead_air_intervals(input_path, dur_s, pre_cfg)
        keep_pre = complement_intervals(removed, dur_s)
        keep_pre = drop_short(keep_pre, 0.050)

        report["prepass_removed"] = removed
        report["prepass_keep"] = keep_pre

        prepass_full = os.path.join(workdir, "prepass_full.flac")
        print(f"[prepass] removing {len(removed)} dead-air intervals; rendering prepass_full.flac", file=sys.stderr)

        ffmpeg_render_keep_intervals(
            input_path,
            keep_pre,
            prepass_full,
            fade_s=args.fade,
            audio_codec="flac",
        )

        # 2) Make a VAD-friendly wav
        prepass_vad = os.path.join(workdir, "prepass_vad_16k_mono.wav")
        prepass_out_s = sum((e - s) for s, e in keep_pre)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-v", "error",
            "-progress", "pipe:2",
            "-i", prepass_full,
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            prepass_vad
        ]
        run_ffmpeg_with_progress(cmd, label="wav", total_out_s=prepass_out_s)

        # 3) Silero VAD keep intervals on prepass audio
        print(f"[vad] running Silero VAD (streaming) on prepass audio", file=sys.stderr)
        vad_keep = silero_vad_speech_segments_streaming(prepass_vad, vad_cfg)
        report["vad_keep"] = vad_keep

        # 4) Render final from prepass_full using VAD keep intervals
        codec, extra = guess_codec_from_ext(output_path)
        print(f"[render] rendering final output -> {output_path}", file=sys.stderr)

        ffmpeg_render_keep_intervals(
            prepass_full,
            vad_keep,
            output_path,
            fade_s=args.fade,
            audio_codec=codec,
            extra_args=extra,
        )

        # 5) Optional JSON report
        if args.report_json:
            ensure_parent_dir(args.report_json)
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[info] wrote report: {args.report_json}", file=sys.stderr)

        print("[done]", file=sys.stderr)

    finally:
        if not args.keep_temp:
            # Clean temp dir
            try:
                workdir_ctx.cleanup()  # type: ignore
            except Exception:
                pass


if __name__ == "__main__":
    main()
