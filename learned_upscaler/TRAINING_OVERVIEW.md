# Learned upscaler — training & smoke (quick reference)

Work from **`comma_video_compression_challenge/`** (this folder’s parent).

## First time (train → zip → quick eval)

```bash
bash learned_upscaler/run_p10_smoke.sh
```

Needs **`videos/0.mkv`**. Writes **`learned_upscaler/micro_upscaler.pt`**, copies into **`p10_neural_infra`**, runs **`compress.sh`** with **`tiny_test.txt`**, then **`evaluate.sh --quick`** (adds **`--device mps`** on macOS).

## Pause (you choose)

| What you want | What to do |
|----------------|------------|
| **Short break, keep RAM** | **Ctrl+Z**, then **`fg`** when back (no disk save). |
| **Stop and save** | **Ctrl+C** during a **batch** → writes **`learned_upscaler/checkpoints/train_resume.pt`**. |
| **Laptop shutdown / SIGTERM** | Between batches, **`train.py`** tries to save the same checkpoint (exit **143**). |

## Resume training

```bash
bash learned_upscaler/run_p10_smoke.sh --resume
```

Use the **same** **`--epochs`**, **`--batch-size`**, **`--loss`**, etc. as before (pass them again; later flags win).

**Temp LR** (`learned_upscaler/temp_lr_train.mkv`): **automatic** — reused if it exists and is **not older** than **`--video`** (mtime). Otherwise ffmpeg runs. **`--force-encode`** always re-encodes; **`--skip-encode`** never encodes (error if temp missing).

## Hard shutdown / lost checkpoint

- Anything **inside the current epoch** that was not written yet is **lost** (no disk flush).
- With default **`--checkpoint-every 1`**, the last **fully finished** epoch is saved to **`checkpoints/train_resume.pt`** after each epoch — so you usually only lose **at most one epoch** of progress.
- If **`train_resume.pt`** is **gone**, you **cannot** resume: start fresh **without** **`--resume`** (or delete the broken checkpoint file if partial).

## Train only (no compress/eval until done)

```bash
uv run python learned_upscaler/train.py --resume --epochs 100 --batch-size 8 --lr 1e-3
```

## Re-score only (training already finished)

```bash
cp learned_upscaler/micro_upscaler.pt submissions/p10_neural_infra/
bash submissions/p10_neural_infra/compress.sh --video-names-file tiny_test.txt
bash evaluate.sh --quick --submission-dir ./submissions/p10_neural_infra --device mps
```

## Useful flags (`train.py`)

| Flag | Role |
|------|------|
| `--resume` | Load **`--checkpoint`** (default `checkpoints/train_resume.pt`). |
| `--checkpoint-every N` | Save after every **N** completed epochs (**default 1**; use **0** to disable). |
| `--force-encode` | Always rebuild temp LR (e.g. after **`--scale`** change). |
| `--skip-encode` | Never encode; temp file must exist. |
| `--loss l1` | Pixel-only (faster than default VGG loss). |
| `--temporal-weight` | **Step 6:** temporal L1 on frame deltas (**default 0.5**); **0** = spatial-only (Step-5-style). |

More detail and experiment log: **`ML_EXPERIMENTS.md`**.
