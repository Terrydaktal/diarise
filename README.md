# diarise

Single-command pipeline for long recordings:

- Speech timestamps (VAD) as JSON
- Optional condensed audio (stitch speech together)
- Whisper transcription (faster-whisper) to TXT + JSON + a sectioned TXT (default 30 min bins)

Recordings and generated outputs are intentionally ignored by git (see `.gitignore`).

## Requirements

- `ffmpeg` + `ffprobe` in `PATH`
- Python 3.10+ (tested with 3.12)

Python packages (install into a venv):

- `numpy`
- `torch` + `torchaudio` (Silero VAD path)
- `webrtcvad` (WebRTC VAD)
- `faster-whisper` (Whisper via CTranslate2)

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install numpy torch torchaudio webrtcvad faster-whisper
```

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
- Whisper device: `--whisper-device cuda` (falls back to CPU if CUDA init fails)
- faster-whisper internal VAD filter: enabled
- Sectioned transcript bins: `--section-minutes 30` (set `0` to disable)

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
2. WebRTC VAD on the prepassed audio
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
- Sectioned transcript anchoring uses the earliest available file timestamp from `stat` (Access/Modify/Change/Birth).
