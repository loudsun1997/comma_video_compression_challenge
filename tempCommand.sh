#!/usr/bin/env bash
# True Phase 8: test_videos.zip → remux .hevc → videos/{0..N}.mkv → p6 compress → evaluate.
#
# The HuggingFace archive holds many **.hevc** streams in **nested paths** (each named e.g.
# video.hevc). **Do not** `unzip -j` into videos/ — every entry would overwrite the same
# basename. This script mirrors **download_and_remux.sh**: unzip with paths preserved, then
# **ffmpeg** remux to **0.mkv, 1.mkv, …** in sorted path order.
#
#   ./tempCommand.sh              # skip download/remux if videos/ already has ≥ MIN_MKVS .mkv
#   ./tempCommand.sh --redownload # wipe .mkv, re-fetch or reuse zip, remux again
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ZIP_URL="https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip"
ZIP_PATH="${ROOT}/test_videos.zip"
EXTRACT_DIR="${ROOT}/test_videos_extract"
MIN_MKVS=60
FPS=20

REDOWNLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --redownload) REDOWNLOAD=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--redownload]"
      echo "  --redownload  remove built .mkv, remux from zip again (re-downloads zip if missing)"
      exit 0 ;;
    *)
      echo "Unknown option: $1 (try --help)" >&2
      exit 2 ;;
  esac
done

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found in PATH" >&2; exit 1; }

count_mkvs() {
  find videos -maxdepth 1 -type f -name '*.mkv' 2>/dev/null | wc -l | tr -d ' '
}

mkdir -p videos
N="$(count_mkvs)"

need_fetch=0
if [[ "$REDOWNLOAD" -eq 1 ]]; then
  need_fetch=1
elif [[ "${N}" -lt "${MIN_MKVS}" ]]; then
  need_fetch=1
fi

if [[ "$need_fetch" -eq 1 ]]; then
  echo "==> Dataset: need full test set (have ${N} .mkv under videos/, want ≥ ${MIN_MKVS})"
  echo "==> Clearing videos/*.mkv and stray videos/*.hevc"
  find videos -maxdepth 1 -type f \( -name '*.mkv' -o -name '*.hevc' \) -delete

  if [[ ! -f "$ZIP_PATH" ]]; then
    echo "==> Download test_videos.zip (~2.4 GB)"
    curl -fL --retry 3 --retry-delay 5 -o "$ZIP_PATH" "$ZIP_URL"
  else
    echo "==> Using existing $(basename "$ZIP_PATH") (delete it to force a new download)"
  fi

  echo "==> Unzip with paths preserved → ${EXTRACT_DIR##*/}/"
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  unzip -q -o "$ZIP_PATH" -d "$EXTRACT_DIR"

  N_HEVC="$(find "$EXTRACT_DIR" -type f -name '*.hevc' | wc -l | tr -d ' ')"
  echo "==> Found ${N_HEVC} .hevc files in archive"
  if [[ "${N_HEVC}" -lt "${MIN_MKVS}" ]]; then
    echo "ERROR: expected at least ${MIN_MKVS} .hevc files after unzip (got ${N_HEVC})." >&2
    exit 1
  fi

  echo "==> Remux to videos/0.mkv … (ffmpeg -f hevc, same as download_and_remux.sh)"
  idx=0
  > "${ROOT}/full_test_video_names.txt"
  while IFS= read -r src; do
    [[ -z "$src" ]] && continue
    rel="${src#"${EXTRACT_DIR}/"}"
    dst="${ROOT}/videos/${idx}.mkv"
    printf '  remux %d/%d → %s.mkv\n' "$((idx + 1))" "$N_HEVC" "$idx"
    ffmpeg -y -loglevel error -f hevc -framerate "$FPS" -r "$FPS" -i "$src" -c copy -metadata segment="$rel" "$dst"
    echo "${idx}.mkv" >> "${ROOT}/full_test_video_names.txt"
    idx=$((idx + 1))
  done < <(find "$EXTRACT_DIR" -type f -name '*.hevc' | LC_ALL=C sort)

  rm -rf "$EXTRACT_DIR"
  echo "==> Remove zip to free disk"
  rm -f "$ZIP_PATH"

  LINES="$idx"
  echo "==> Roster: ${LINES} clips → full_test_video_names.txt"
  if [[ "${LINES}" -lt "${MIN_MKVS}" ]]; then
    echo "ERROR: remux produced only ${LINES} files (need ≥ ${MIN_MKVS})." >&2
    exit 1
  fi
else
  echo "==> Dataset: skip download/remux (already ${N} .mkv files under videos/)"
  echo "==> Rebuild full_test_video_names.txt from videos/*.mkv"
  ls -1 videos/ | grep '\.mkv$' | LC_ALL=C sort -V > full_test_video_names.txt
  LINES="$(wc -l < full_test_video_names.txt | tr -d ' ')"
  echo "==> Roster line count: ${LINES}"
  if [[ "${LINES}" -lt "${MIN_MKVS}" ]]; then
    echo "ERROR: expected at least ${MIN_MKVS} lines in full_test_video_names.txt (got ${LINES})." >&2
    echo "Run: $0 --redownload" >&2
    exit 1
  fi
fi

echo "==> Compress (p6; parallel files; JOBS = logical CPUs in compress.sh)"
bash submissions/p6_bicubic_c27_ft1/compress.sh --video-names-file full_test_video_names.txt

echo "==> Evaluate (same roster)"
EVAL_EXTRA=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  EVAL_EXTRA=(--device mps)
fi
bash evaluate.sh --submission-dir ./submissions/p6_bicubic_c27_ft1 --video-names-file full_test_video_names.txt "${EVAL_EXTRA[@]}"

echo "==> Done. Report: submissions/p6_bicubic_c27_ft1/report.txt"
