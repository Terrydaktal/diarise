# diarise

Utilities for turning long recordings into:

- Speech timestamps (VAD) in JSON
- Optional condensed audio built by stitching speech regions together
- Whisper transcription (faster-whisper) with timestamps
- A “sectioned” transcript grouped into half-hour windows, anchored to the original file’s filesystem time

This repo intentionally **does not** track recordings or generated outputs in git.

## Requirements

- `ffmpeg` + `ffprobe` in `PATH`
- Python 3.10+ (tested with 3.12)

Python packages (installed into a venv):

- `torch` + `torchaudio` (needed for Silero VAD path in `diarise`)
- `webrtcvad` (needed for WebRTC VAD)
- `faster-whisper` (Whisper transcription via CTranslate2)

## Setup

Create a venv in this repo (the scripts assume `.venv` is fine):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install torch torchaudio webrtcvad faster-whisper
```

## `diarise` (All-In-One)

`./diarise` is the main entry point. It can:

- Output speech timestamps (VAD) to JSON
- Render condensed audio from VAD keep-intervals
- Transcribe to TXT/JSON with Whisper (faster-whisper)

The `./diarise` script can:

1. Produce only speech timestamps (JSON), fast:

```bash
./diarise Recording.mp3 --timestamps-only --vad-backend webrtc --report-json Recording_webrtc_vad.json
```

2. Render a condensed audio file from an existing VAD JSON:

```bash
./diarise Recording.mp3 Recording_condensed.m4a --condense-from-json Recording_webrtc_vad.json
```

Notes:

- `--timestamps-only` **skips** the old “prepass” dead-air removal and does not render any audio.
- `--condense-from-json` preserves the input’s sample rate and channel count in the output.

### Whisper transcription (TXT + JSON)

Transcribe an input to `<out_prefix>.txt` and `<out_prefix>.json`:

```bash
. .venv/bin/activate
./diarise Recording.mp3 --transcribe --whisper-model base --whisper-device cpu --whisper-compute-type int8
```

By default, this uses faster-whisper’s built-in `vad_filter` (enabled) and does **not** do any external preprocessing.

If you want a fast prepass to skip silence, add `--prevad` (WebRTC VAD) which feeds `clip_timestamps` to Whisper:

```bash
./diarise Recording.mp3 --transcribe --prevad --whisper-model base --whisper-device cpu --whisper-compute-type int8
```

## `whisper_transcribe.py` (Optional)

Transcribe directly (default: faster-whisper built-in VAD filter enabled):

```bash
. .venv/bin/activate
python whisper_transcribe.py Recording.mp3 --model base --device cpu --compute-type int8
```

Outputs:

- `<out_prefix>.txt`
- `<out_prefix>.json` (includes per-segment `start`/`end` in seconds)
- `<out_prefix>_sectioned.txt` (default bins = 30 minutes; disable with `--section-minutes 0`)

### `--prevad` mode (WebRTC VAD prepass + preprocessing outputs)

`--prevad` does two things:

1. Runs WebRTC VAD first and feeds the resulting keep-intervals to Whisper via `clip_timestamps`.
2. Writes preprocessing outputs to make the timeline mapping explicit:
   - `<out_prefix>_prevad.json` (contains `vad_keep` and a `timeline_map`)
   - `<out_prefix>_prevad_condensed.m4a` (stitched audio built from `vad_keep`)

Example:

```bash
python whisper_transcribe.py Recording.mp3 --model base --device cpu --compute-type int8 --prevad
```

If you want `--prevad` but **don’t** want the extra files:

```bash
python whisper_transcribe.py Recording.mp3 --model base --prevad --prevad-no-preprocess
```

### Preserving original timestamps when transcribing condensed audio

Condensed audio has its own (shorter) timeline. To convert Whisper segment times back to the original recording timeline,
`--prevad` writes a `timeline_map` into `<out_prefix>_prevad.json`.

When you later transcribe the condensed file:

- If the input filename ends with `_prevad_condensed.*`, `whisper_transcribe.py` will auto-detect the sibling
  `<stem>_prevad.json` and automatically remap timestamps.
- Otherwise, pass the map explicitly with `--timeline-map`.

Example (auto-detect mapping):

```bash
python whisper_transcribe.py Recording_prevad_condensed.m4a --model base --device cpu --compute-type int8
```

Example (explicit mapping):

```bash
python whisper_transcribe.py some_condensed.m4a --model base --timeline-map Recording_prevad.json
```

In the output JSON:

- `.segments` are the remapped segments in original time (when a map is used)
- `.segments_condensed` holds the raw segments in condensed time (debug)

## Sectioned transcript with real timestamps

`whisper_transcribe.py` also writes `<out_prefix>_sectioned.txt` (default: 30-minute windows).

The “real time” anchor is taken from:

1. `--anchor-file` (highest priority), else
2. `--diarise-report` (uses its `input`), else
3. If transcribing a `_condensed` file: best-effort guessing of the matching report, else
4. The input file itself

It uses filesystem **birth time** if available; otherwise falls back to **mtime**.

## GPU (CUDA)

To try GPU inference (only works if your CUDA/NVIDIA stack is correctly installed and accessible):

```bash
python whisper_transcribe.py Recording.mp3 --model base --device cuda --compute-type float16
```

If CUDA init fails, re-run with `--device cpu`.

## Progress / “nothing is happening”

- `diarise` prints VAD progress in hours processed.
- `whisper_transcribe.py` prints `[progress]` lines based on decoded segment end times; use `--log-every` to control frequency.

## Git hygiene

- `.gitignore` excludes common recording formats and all generated outputs (`.json`, `.txt`, condensed audio, logs, etc).
- If you accidentally added recordings to git in the past, rewrite history before pushing.
