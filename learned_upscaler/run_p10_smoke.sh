#!/usr/bin/env bash
# Train TS-SPCN (default: Step 5 VGG perceptual + L1) on videos/0.mkv, copy weights into
# p10_neural_infra, build archive.zip for tiny_test.txt, run evaluate.sh --quick.
#
# Requires: videos/0.mkv, uv + project deps (torchvision for --loss vgg).
#
#   bash learned_upscaler/run_p10_smoke.sh
#   bash learned_upscaler/run_p10_smoke.sh --loss l1 --epochs 2
#   bash learned_upscaler/run_p10_smoke.sh --epochs 100 --batch-size 4
#
# Colab (no uv): see learned_upscaler/run_p10_smoke_colab.sh and run_p10_smoke_colab.ipynb
#
# Extra CLI args are forwarded to learned_upscaler/train.py (later flags override earlier).
# Pause: Ctrl+C during a batch saves checkpoints/train_resume.pt — then e.g.:
#   bash learned_upscaler/run_p10_smoke.sh --resume
# (train.py auto-reuses temp_lr_train.mkv if it exists and is not older than the source video.)
# Hard shutdown: default --checkpoint-every 1 keeps last completed epoch on disk; see TRAINING_OVERVIEW.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found in PATH" >&2; exit 1; }

if [[ ! -f videos/0.mkv ]]; then
  echo "ERROR: videos/0.mkv missing (training clip). Add the clip or run tempCommand.sh to populate videos/." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
uv run python learned_upscaler/train.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-3 \
  "$@"

cp learned_upscaler/micro_upscaler.pt submissions/p10_neural_infra/

bash submissions/p10_neural_infra/compress.sh --video-names-file tiny_test.txt

EVAL_EXTRA=()
if [[ "$(uname -s)" == Darwin ]]; then
  EVAL_EXTRA=(--device mps)
fi
bash evaluate.sh --quick --submission-dir ./submissions/p10_neural_infra "${EVAL_EXTRA[@]}"

echo "==> Done. Report: submissions/p10_neural_infra/report.txt"
