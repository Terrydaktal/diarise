# diarise

Single-command pipeline for long recordings:

- Speech timestamps (VAD) as JSON
- Optional condensed audio (stitch speech together)
- Whisper transcription (via `faster-whisper`) to TXT + JSON + sectioned/timestamped TXT (default 30 min bins)

Recordings and generated outputs are intentionally ignored by git (see `.gitignore`).

## Requirements

- `ffmpeg` + `ffprobe` in `PATH`
- Python 3.10+ (tested with 3.12)

Python packages (install into a venv):

- `numpy`
- `torch` + `torchaudio` (Silero VAD path)
- `webrtcvad` (WebRTC VAD)
- `faster-whisper` (Whisper via CTranslate2)
- `nvidia-cublas-cu12` (recommended for CUDA; provides `libcublas.so.12`)

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install numpy torch torchaudio webrtcvad faster-whisper nvidia-cublas-cu12
```

Notes:

- `./diarise` is a small wrapper that runs `diarise.py` using `./.venv/bin/python` (or `./venv/bin/python`).
- It also auto-adds `.../site-packages/nvidia/*/lib` to `LD_LIBRARY_PATH` so CTranslate2 can find CUDA libs like `libcublas.so.12`.

## Quick Start (Transcribe)

Default behavior is transcription when you provide only an input:

```bash
. .venv/bin/activate
./diarise Recording.mp3
```

This writes:

- `Recording_whisper_base.txt`
- `Recording_whisper_base.json`
- `Recording_whisper_base_sectioned.txt`
- `Recording_whisper_base_timestamped.txt`

Defaults:

- Whisper model: `--whisper-model base`
- Whisper device: `--whisper-device cuda` (does not fall back to CPU; pass `--whisper-device cpu` to force CPU)
- Whisper language: `--whisper-language en`
- Whisper compute type: `--whisper-compute-type float16` (CPU falls back to `int8`)
- faster-whisper internal VAD filter: enabled
- Sectioned transcript bins: `--section-minutes 30` (set `0` to disable)

Use a different model (e.g. `small`):

```bash
./diarise Recording.mp3 --whisper-model small
```

Force CPU:

```bash
./diarise Recording.mp3 --whisper-device cpu --whisper-compute-type int8
```

Disable sectioned output:

```bash
./diarise Recording.mp3 --section-minutes 0
```

## Faster Skip-Silence Transcribe (`--prevad`)

Add `--prevad` to run:

1. Dead-air prepass removal (stitch “keep” intervals into a temp file)
2. WebRTC VAD on a streamed ffmpeg concat of kept intervals (no full re-encode)
3. Map VAD intervals back to the original timeline
4. Whisper on the original input using `clip_timestamps` (disables whisper VAD)

```bash
./diarise Recording.mp3 --prevad
```

If you need to inspect intermediates:

```bash
./diarise Recording.mp3 --prevad --keep-temp
```

## Batched Transcription (`--batched`)

Enable faster-whisper’s `BatchedInferencePipeline` with `batch_size=16`:

```bash
./diarise Recording.mp3 --batched
```

It can be combined with `--prevad`:

```bash
./diarise Recording.mp3 --prevad --batched
```

Notes:

- Batched mode requires either Whisper VAD (`--whisper-vad-filter`, default) or `--prevad` (clip timestamps). If you disable VAD in batched mode, diarise will retry non-batched.
- Batched mode decodes the full audio into memory (so very long recordings can be RAM-heavy).

## Postprocess Section Files (`--postprocess-json`)

If you already have `*_whisper_*.json`, you can regenerate the sectioned + timestamped files without re-running Whisper:

```bash
./diarise Recording.mp3 --postprocess-json Recording_whisper_base.json
```

## VAD Timestamps Only (JSON)

Write VAD keep intervals without rendering audio:

```bash
./diarise Recording.mp3 --timestamps-only --vad-backend webrtc --report-json Recording_webrtc_vad.json
```

Notes:

- `--timestamps-only` skips the dead-air prepass and audio rendering.
- Output JSON contains `vad_keep` as `[[start,end], ...]` in seconds (original timeline).

## Condensed Audio

Condense with the older full pipeline (prepass + Silero VAD + render) by providing an output path:

```bash
./diarise Recording.mp3 Recording_condensed.m4a --report-json Recording_report.json
```

Or render condensed audio from an existing VAD JSON:

```bash
./diarise Recording.mp3 Recording_condensed.m4a --condense-from-json Recording_webrtc_vad.json
```

`--condense-from-json` preserves the input sample rate and channel count in the output.

## Progress

- `--timestamps-only` prints VAD progress in hours processed.
- Whisper transcription prints progress approximately by decoded segment end time. Control frequency with `--whisper-log-every` (set `0` to disable).

## Repository Notes

- `.hf_cache/` is used for HuggingFace downloads to avoid permission issues with `~/.cache`.
- If CUDA is not usable on your machine, use `--whisper-device cpu`.
- Timestamped/sectioned transcript anchoring uses the earliest available file timestamp from `stat` (Access/Modify/Change/Birth) on the original input audio file,
  and treats it as the end-of-recording timestamp (computes start time as `anchor - duration`).
- Anchor metadata is written into `*_whisper_*.json` as `anchor_file`, `anchor_time`, `anchor_time_src`, and `anchor_time_assumed=end_of_recording`.
