#!/usr/bin/env bash
# Train TS-SPCN (default: Step 5 VGG perceptual + L1) on videos/0.mkv, copy weights into
# p10_neural_infra, build archive.zip for public_test_video_names.txt, run evaluate.sh.
#
# Requires: videos/0.mkv, project Python deps. Uses `uv run python` if uv is installed,
# else `python3` (e.g. Google Colab after `pip install` deps; see run_p10_smoke_colab.ipynb).
#
#   bash learned_upscaler/run_p10_smoke.sh
#   bash learned_upscaler/run_p10_smoke.sh --loss l1 --epochs 2
#   bash learned_upscaler/run_p10_smoke.sh --epochs 100 --batch-size 4
#
# Colab: Runtime → GPU, open run_p10_smoke_colab.ipynb, or: pip install -q … then run this same command.
#
# Extra CLI args are forwarded to learned_upscaler/train.py (later flags override earlier).
# Pause: Ctrl+C during a batch saves checkpoints/train_resume.pt — then e.g.:
#   bash learned_upscaler/run_p10_smoke.sh --resume
# (train.py auto-reuses temp_lr_train.mkv if it exists and is not older than the source video.)
# Hard shutdown: default --checkpoint-every 1 keeps last completed epoch on disk; see TRAINING_OVERVIEW.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  RUN_TRAIN=(uv run python)
else
  command -v python3 >/dev/null 2>&1 || { echo "ERROR: need uv or python3 in PATH" >&2; exit 1; }
  RUN_TRAIN=(python3)
fi

if [[ ! -f videos/0.mkv ]]; then
  echo "ERROR: videos/0.mkv missing (training clip). Add the clip or run tempCommand.sh to populate videos/." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
"${RUN_TRAIN[@]}" learned_upscaler/train.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-3 \
  "$@"

cp learned_upscaler/micro_upscaler.pt submissions/p10_neural_infra/

bash submissions/p10_neural_infra/compress.sh --video-names-file public_test_video_names.txt

EVAL_EXTRA=()
if command -v nvidia-smi >/dev/null 2>&1; then
  EVAL_EXTRA=(--device cuda)
elif [[ "$(uname -s)" == Darwin ]]; then
  EVAL_EXTRA=(--device mps)
fi
bash evaluate.sh --submission-dir ./submissions/p10_neural_infra "${EVAL_EXTRA[@]}"

echo "==> Done. Report: submissions/p10_neural_infra/report.txt"
