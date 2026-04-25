#!/usr/bin/env bash
# p10: 25% bicubic scale + p6-style x265; bundle TS-SPCN weights in archive.zip for neural inflate.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir) IN_DIR="${2%/}"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

if [[ ! -f "${HERE}/micro_upscaler.pt" ]]; then
  echo "ERROR: ${HERE}/micro_upscaler.pt missing. Run: uv run python ${HERE}/model.py" >&2
  exit 1
fi

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR

echo "Parallel file jobs (xargs -P): ${JOBS}"

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -c '
  rel="$1"
  [[ -z "$rel" ]] && exit 0
  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  echo "→ ${IN}  →  ${OUT}"
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=trunc(iw*0.25/2)*2:trunc(ih*0.25/2)*2:flags=bicubic" \
    -c:v libx265 -preset slower -crf 27 \
    -g 60 -bf 0 -x265-params "keyint=60:min-keyint=1:scenecut=40:no-sao=1:frame-threads=1:log-level=warning" \
    -r 20 "$OUT"
' _ {}

cd "$ARCHIVE_DIR"
cp "${HERE}/model.py" .
cp "${HERE}/micro_upscaler.pt" .
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip (includes model.py + micro_upscaler.pt)"
