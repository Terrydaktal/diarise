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

    dt = time.time() - t0
    print(f"[done] wrote {out_txt} and {out_json} in {dt/60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
