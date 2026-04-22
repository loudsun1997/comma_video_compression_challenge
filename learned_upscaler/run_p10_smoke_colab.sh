#!/usr/bin/env bash
# Colab-friendly smoke: same steps as run_p10_smoke.sh, but uses python3 (not uv).
# Use a GPU runtime in Colab; evaluation uses --device cuda when nvidia-smi exists.
#
#   cd <comma_video_compression_challenge>   # root with videos/, frame_utils.py, tiny_test.txt
#   bash learned_upscaler/run_p10_smoke_colab.sh
#   bash learned_upscaler/run_p10_smoke_colab.sh --loss l1 --epochs 2
#
# Colab: after opening run_p10_smoke_colab.ipynb (or manual pip install), run:
#   !bash learned_upscaler/run_p10_smoke_colab.sh
#
# Optional: PATH=/usr/bin:$PATH and PYTHON=python3 if you use a custom venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON} not found in PATH" >&2
  exit 1
fi

if [[ ! -f videos/0.mkv ]]; then
  echo "ERROR: videos/0.mkv missing (training clip). Upload it under videos/0.mkv." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
"$PYTHON" learned_upscaler/train.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-3 \
  "$@"

cp learned_upscaler/micro_upscaler.pt submissions/p10_neural_infra/

bash submissions/p10_neural_infra/compress.sh --video-names-file tiny_test.txt

EVAL_EXTRA=()
if command -v nvidia-smi >/dev/null 2>&1; then
  EVAL_EXTRA=(--device cuda)
elif [[ "$(uname -s)" == Darwin ]]; then
  EVAL_EXTRA=(--device mps)
fi
bash evaluate.sh --quick --submission-dir ./submissions/p10_neural_infra "${EVAL_EXTRA[@]}"

echo "==> Done. Report: submissions/p10_neural_infra/report.txt"
