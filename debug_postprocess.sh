#!/usr/bin/env bash
set -euo pipefail

AUDIO=${1:-"/media/lewis/1b/Diary Recordings/Recording_1828.mp3"}
JSON=${2:-"$PWD/Recording_1828_whisper_large-v3.json"}
BIN_MINUTES=${3:-30}
LOGDIR=${4:-"$PWD/debug_postprocess_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$LOGDIR"
set -x

date -Is
uname -a
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
ls -lh "$AUDIO" "$JSON"
stat "$AUDIO" "$JSON"

echo "== ffprobe duration check =="
timeout --foreground -k 5s 60s ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$AUDIO" \
  | tee "$LOGDIR/ffprobe_duration.txt"
echo "ffprobe_rc=${PIPESTATUS[0]}"

echo "== ffmpeg tail decode check (same pattern as diarise) =="
set +e
timeout --foreground -k 5s 120s ffmpeg -nostdin -hide_banner -v error -sseof -60 -t 10 -i "$AUDIO" -vn -f null - \
  >"$LOGDIR/ffmpeg_tail.stdout.log" 2>"$LOGDIR/ffmpeg_tail.stderr.log"
TAIL_RC=$?
set -e
echo "ffmpeg_tail_rc=$TAIL_RC" | tee "$LOGDIR/ffmpeg_tail.rc"

if command -v strace >/dev/null && [[ "$TAIL_RC" -eq 124 ]]; then
  echo "== tail decode timed out; collecting strace sample =="
  set +e
  timeout 30s strace -f -tt -o "$LOGDIR/ffmpeg_tail.strace" \
    ffmpeg -hide_banner -v error -sseof -60 -t 10 -i "$AUDIO" -vn -f null - >/dev/null 2>&1
  echo "strace_rc=$?" | tee "$LOGDIR/ffmpeg_tail.strace.rc"
  set -e
fi

echo "== diarise postprocess only (no transcription) =="
set +e
timeout --foreground -k 5s 600s ./diarise "$AUDIO" --postprocess-json "$JSON" --section-minutes "$BIN_MINUTES" \
  >"$LOGDIR/postprocess.stdout.log" 2>"$LOGDIR/postprocess.stderr.log"
POST_RC=$?
set -e
echo "postprocess_rc=$POST_RC" | tee "$LOGDIR/postprocess.rc"

OUT_PREFIX="${JSON%.*}"
ls -lh "${OUT_PREFIX}_sectioned.txt" "${OUT_PREFIX}_timestamped.txt" 2>/dev/null || true
echo "LOGDIR=$LOGDIR"
