# ML phase — learned upscaler (TS-SPCN)

**Goal:** Train a tiny super-resolution head (bottleneck + low-res residual stack + **PixelShuffle**) so submissions can encode at **~25%** linear scale (4× upscale), beat the **size trap** (weights ≪ rate penalty), and later align reconstructions with **SegNet-sensitive** structure via perceptual loss.

**Constraint reminder:** Anything shipped in **`archive.zip`** counts toward **rate**; keep checkpoints **well under ~100 KB** unless the bitrate savings justify more.

**Code layout:**

| Path | Role |
|------|------|
| `learned_upscaler/model.py` | `TS_SPCN`, `ResidualBlock`; run as script to print param count + write `micro_upscaler.pt` |
| `learned_upscaler/run_p10_smoke.sh` | **One-shot:** train (default VGG loss) → copy `micro_upscaler.pt` → `p10` compress (`tiny_test.txt`) → `evaluate.sh --quick` |
| `learned_upscaler/TRAINING_OVERVIEW.md` | Commands: first run, resume, re-eval, flags; hard shutdown / lost checkpoint |
| Pause / resume | **`train.py`:** **Ctrl+C** (during a batch) saves `learned_upscaler/checkpoints/train_resume.pt`; **SIGTERM/SIGHUP** also save **between batches**. Continue with **`--resume`**. Default **`--checkpoint-every 1`** re-saves after **each finished epoch** (limits loss on hard shutdown). **Temp LR:** **auto** reuse if present and not older than `--video` (mtime); **`--force-encode`** / **`--skip-encode`** override. **Ctrl+Z** / **`fg`** = RAM suspend, no disk. Cheat sheet: **`learned_upscaler/TRAINING_OVERVIEW.md`**. |
| `learned_upscaler/tests/` | `unittest` checks for shape + checkpoint size |
| This file | Training / infra / ablation log |

---

## Step 1 — architecture footprint (verified)

Run from **`comma_video_compression_challenge/`**:

```bash
uv run python learned_upscaler/model.py
python -m unittest learned_upscaler.tests.test_model_size -v
```

| Metric | Value (default `upscale_factor=4`, `num_residual_blocks=4`) |
|--------|----------------------------------------------------------------|
| **Parameters** | **18,520** |
| **`micro_upscaler.pt` (state_dict)** | **~79.7 KB** on disk (PyTorch pickle; under **100 KB** ✓) |

*Note:* Earlier back-of-envelope **~11.6k** params assumed a smaller mapping stack; this exact spec is **18,520** — still tiny vs video bitrate.

**Next branches (pick one):**

1. **Infrastructure:** See **Step 2** — `submissions/p10_neural_infra/` (done for smoke test).
2. **Training:** Dataset loaders from **`videos/`** or comma2k19, **L1/L2 + SegNet feature loss**, export `state_dict` into submission zip.

---

## Step 2 — `p10_neural_infra` (speed / zip smoke)

**Submission:** `submissions/p10_neural_infra/` — **25%** bicubic encode (vs p6 **40%**), same x265 recipe; **`archive.zip`** includes **`0.mkv`…** plus **`model.py`** + **`micro_upscaler.pt`**. **`inflate.py`** runs **`TS_SPCN`** (4×) on GPU (`cuda` / **`mps`** / `cpu`) with bilinear fallback to exact **`camera_size`**.

**Ensure weights exist** (once per clone):

```bash
uv run python submissions/p10_neural_infra/model.py
```

**Tiny harness** (from repo root):

```bash
bash submissions/p10_neural_infra/compress.sh --video-names-file tiny_test.txt
bash evaluate.sh --quick --submission-dir ./submissions/p10_neural_infra --device mps
```

**Measured (Apple Silicon, MPS, `0.mkv` only, Apr 2026):**

| Stage | Wall time (approx.) |
|--------|----------------------|
| `compress.sh` (1 clip, `slower` x265) | ~**29 s** |
| `evaluate.sh --quick` (unzip + neural inflate 1200 f + **600** eval batches) | ~**84 s** |

**Report (`tiny_test.txt`):**

| | Value |
|--|--------|
| Zip (incl. weights) | 632,322 bytes |
| Rate | 0.01684203 |
| PoseNet / SegNet | **~160 / ~0.50** (untrained model — **ignore** for quality) |
| **Score** | **~90.9** (expected garbage until training) |

**Speed trap note:** `inflate.sh` spawns **one Python process per video**, so the **TS-SPCN** loads **once per clip**. For **64** clips on a **T4**, if wall time scales roughly linearly with work, budget **inflate + eval** against the **30-minute** server limit; next optimization is a **single-process** inflater over the whole roster (or fewer processes) to cut redundant loads and startup.

---

## Log

| Date | Experiment | Notes |
|------|--------------|--------|
| 2026-04 | `p10_neural_infra` + `tiny_test` + MPS | Infra OK; ~29 s compress, ~84 s full `--quick` eval; rate **~0.0168**; score trash until training. |
| 2026-04 | **Step 3** L1 overfit `0.mkv` + redeploy `p10` | `train.py` **100** epochs, **bs=8**, MPS, **~43 min**; eval score **3.42** (was **~90.9** random). See **Step 3** below. |
| 2026-04 | **Step 4** compression-aware `train.py` | ffmpeg **p10** chain → `temp_lr_train.mkv`; **yuv420** load; **AdamW** + **L1**; tqdm + `--no-progress`. **1-epoch smoke OK** (~66 s incl. encode). |
| 2026-04 | **Step 4** full **100** ep + `p10` smoke | **~31 min** train; **SegNet 0.00824** (beats bicubic **0.00945**); **PoseNet 0.89** (regression); **score 4.24**. |
| 2026-04 | **Step 5** VGG perceptual (default **`--loss vgg`**) + **`p10` smoke** | **100** ep (incl. resume), **bs=8**, **λ=0.1**; **`tiny_test` / MPS**; SegNet **0.00820**, PoseNet **0.990**, score **4.39** — see **Step 5** table. |
| 2026-04 | **Step 6** temporal + VGG (`train.py`) | Default **`--temporal-weight 0.5`**; **`0`** = spatial-only. Log metrics in **Step 6** table after smoke run. |

---

## Step 3 — L1 overfit on `0.mkv` (accuracy probe)

**Script:** `learned_upscaler/train.py` — loads **`videos/0.mkv`**, builds **25% bicubic** low-res (even dims, matches compress intent), trains **TS-SPCN** with **L1** vs full-res RGB, writes **`learned_upscaler/micro_upscaler.pt`**.

From **`comma_video_compression_challenge/`**:

```bash
PYTHONUNBUFFERED=1 uv run python learned_upscaler/train.py --epochs 100 --batch-size 8 --lr 1e-3
```

Uses **`tqdm`**: decode progress, per-epoch batch bar, outer epoch bar with **L1** postfix. Pass **`--no-progress`** for plain logs (CI / redirected output).

- **`--batch-size 4`** is gentler on memory; **8** was used for a faster 100-epoch run (~**43 min** wall on MPS in one session).
- After training, **refresh both** the tree checkpoint **and** the submission zip (rate uses **`archive.zip`**):

```bash
cp learned_upscaler/micro_upscaler.pt submissions/p10_neural_infra/
bash submissions/p10_neural_infra/compress.sh --video-names-file tiny_test.txt
bash evaluate.sh --quick --submission-dir ./submissions/p10_neural_infra --device mps
```

**Last logged result (trained vs random):**

| | Random init | After L1 overfit (100 ep, bs 8) |
|--|-------------|----------------------------------|
| **Score** | **~90.9** | **~3.42** |
| PoseNet | ~160 | ~0.42 |
| SegNet | ~0.50 | ~0.0095 |
| Rate | ~0.01684 | ~0.01688 |

**Caveat (historical):** Step 3 used **synthetic** bicubic lows only. **Default `train.py` is now Step 4 (compression-aware)** — see below.

---

## Step 4 — Compression-aware training (close train/serve gap)

**Problem:** Bicubic-downscaled tensors in RAM ≠ **libx265** artifacts the model sees in **`inflate.py`**.

**Fix:** Before training, **`train.py`** runs **`ffmpeg`** with the **same** arguments as **`submissions/p10_neural_infra/compress.sh`** (25% bicubic scale, **`slower` CRF 27**, GOP 60, `bf=0`, `no-sao`, `frame-threads=1`, **20 fps**) into **`learned_upscaler/temp_lr_train.mkv`**. Targets load from the original **`0.mkv`**; inputs load from that **HEVC** file. Decoding uses **`yuv420_to_rgb`** (same path as eval/inflate), not `rgb24`.

```bash
PYTHONUNBUFFERED=1 uv run python learned_upscaler/train.py --epochs 100 --batch-size 8 --lr 1e-3
```

| Flag | Meaning |
|------|---------|
| `--scale 0.25` | Match p10 (default) |
| `--temp-lr PATH` | Low-res mkv location (default under `learned_upscaler/`) |
| *(default)* | **Auto temp LR:** reuse `--temp-lr` if present and **mtime ≥** `--video`; else encode |
| `--skip-encode` | Never encode; require existing `--temp-lr` |
| `--force-encode` | Always run ffmpeg (overwrite temp), e.g. after changing `--scale` |
| `--keep-temp-lr` | Keep the temp mkv after run |
| `--resume` | Load `checkpoints/train_resume.pt` (or `--checkpoint`) |
| `--checkpoint PATH` | Pause/resume state file (default `checkpoints/train_resume.pt`) |
| `--checkpoint-every N` | Save checkpoint every **N** epochs (**default 1**; **0** = off) |
| `--no-progress` | Disable tqdm |

**Optimizer:** **AdamW** + **L1** (as in your spec). Temp file is **gitignored** (`temp_lr_train.mkv`).

After training, same deploy as Step 3: **`cp`** weights → **`compress.sh`** → **`evaluate.sh --quick`**.

**Logged run (100 epochs, bs=8, AdamW, MPS, `tiny_test` / `0.mkv`):**

| Stage | Wall time (approx.) |
|--------|----------------------|
| `ffmpeg` temp LR + **100 epochs** | **~31 min** total (training tqdm **~30 min**; **L1_avg ≈ 0.0136** last epoch) |
| `compress` + `--quick` eval | ~**2 min** |

| Metric | Bicubic-only L1 (Step 3) | **Compression-aware L1 (Step 4)** | p6 traditional |
|--------|---------------------------|-------------------------------------|----------------|
| **SegNet** | 0.00945386 | **0.00824200** | 0.005784 |
| **PoseNet** | 0.42207003 | **0.89453948** | 0.192 |
| **Rate** | 0.01687519 | 0.01687735 | 0.0324 |
| **Score** | **3.42** | **4.24** | 2.77 |

**Readout:** **SegNet** beat the bicubic-only baseline (**0.00824** vs **0.00945**) — better alignment with what **SegNet** sees when train inputs match **x265**. **PoseNet** regressed vs bicubic-only (**0.89** vs **0.42**), so **overall score** worsened (**4.24** vs **3.42**). Plain **L1** encourages soft edges and frame-wise hallucinations (**temporal jitter**); **Step 5** adds **VGG16 feature loss** to sharpen structure and stabilize motion cues.

---

## Step 5 — VGG perceptual loss (default)

**Idea:** Frozen **VGG16** through **relu2_2** (`features[:9]`) on **ImageNet-normalized** inputs; **MSE** on activations encourages edges and shapes vs pixel **L1** alone. Training objective:

`loss = L1(sr, hr) + λ * MSE(VGG(sr), VGG(hr))` with default **λ = 0.1** (`--vgg-feature-weight`).

**Script:** same **`train.py`** as Step 4; default **`--loss vgg`**. Pixel-only ablation: **`--loss l1`**.

**One command (from `comma_video_compression_challenge/`):**

```bash
bash learned_upscaler/run_p10_smoke.sh
```

Overrides (appended to `train.py`; duplicate flags — last wins):

```bash
bash learned_upscaler/run_p10_smoke.sh --loss l1
bash learned_upscaler/run_p10_smoke.sh --epochs 2 --batch-size 4
```

| Flag | Meaning |
|------|---------|
| `--loss vgg` \| `l1` | **vgg** (default): L1 + VGG feature term; **l1**: Step-4-style pixel only |
| `--vgg-feature-weight` | **λ** on feature MSE (default **0.1**) |

Requires **`torchvision`** (e.g. **`uv sync --group mps`** from repo root). Expect **~12–15 min/epoch** on MPS for full-res VGG (long 100-epoch runs; use **`--resume`** / **`--checkpoint-every 1`** as needed).

The script runs **copy → compress → eval** after training. If you train by hand, repeat those three steps from **Step 3** / **Step 4** docs; then log **SegNet / PoseNet / score** in the table below.

**Logged run (100 epochs, bs=8, AdamW, default VGG loss, compression-aware LR, `tiny_test` / `0.mkv`, MPS, Apr 2026):**

| | Value |
|--|--------|
| Zip (incl. weights) | **633,709** bytes |
| **Rate** | **0.01687897** |
| **SegNet** | **0.00820047** |
| **PoseNet** | **0.98962831** |
| **Score** | **4.39** |

| Metric | Compression-aware **L1** (Step 4) | **VGG perceptual** (Step 5) |
|--------|-------------------------------------|-----------------------------|
| **SegNet** | 0.00824200 | **0.00820047** (slightly better) |
| **PoseNet** | 0.89453948 | **0.98962831** (worse) |
| **Rate** | 0.01687735 | 0.01687897 |
| **Score** | **4.24** | **4.39** (worse) |

**Readout:** **VGG** nudged **SegNet** marginally vs Step-4 **L1**, but **PoseNet** distortion **increased** (still near **~1.0**, i.e. very poor motion / temporal consistency vs the **p6** PoseNet column in Step 4). The combined **score** moved **up** (**worse**) vs Step 4. Perceptual loss on **ImageNet** features is only a partial proxy for comma’s **PoseNet** / **SegNet**; next levers could include **temporal** loss, lighter **λ**, different VGG layers, or **shorter** runs to probe the tradeoff curve.

| Date | Experiment | SegNet | PoseNet | Score | Notes |
|------|------------|--------|---------|-------|-------|
| 2026-04 | Step 5 VGG default (100 ep, smoke) | **0.00820047** | **0.98962831** | **4.39** | `run_p10_smoke.sh`; resume OK; see table above. |
| | Step 6 temporal default (100 ep, smoke) | *TBD* | *TBD* | *TBD* | Fill after `run_p10_smoke.sh` with default **`--temporal-weight 0.5`**. |

---

## Step 6 — Temporal consistency loss (PoseNet-oriented)

**Problem:** **VGG** is **static** (ImageNet); **SISR** treats each frame alone, so sharp lane hallucinations can **jitter** frame-to-frame. **PoseNet** sees that as huge **motion** error.

**Fix:** Penalize mismatch between **temporal differences** of SR vs HR:

\[
\mathcal{L}_{\mathrm{temp}} = \big\| (SR_t - SR_{t-1}) - (HR_t - HR_{t-1}) \big\|_1
\]

(equivalent to matching **frame deltas** in L1). Index **0** uses **duplicate** \(t{-}1=t\) so the term vanishes there.

**Training:** One forward on **`cat(lr_t, lr_{t-1})`** → split **`sr_t` / `sr_{t-1}`**; **`spatial_loss = criterion(sr_t, hr_t)`** (same VGG+L1 or L1 as Step 5); **`total = spatial + temporal_weight * temporal_loss`**. Default **`--temporal-weight 0.5`**.

```bash
bash learned_upscaler/run_p10_smoke.sh
# Spatial-only ablation (Step 5 behavior):
# bash learned_upscaler/run_p10_smoke.sh --temporal-weight 0
```

| Flag | Meaning |
|------|---------|
| `--temporal-weight` | Scale on temporal L1 (**default 0.5**); **0** disables |

**Memory:** Effective **2×** samples per forward vs spatial-only; if MPS OOM, try **`--batch-size 4`**.

**Checkpoint meta** includes **`temporal_weight`**; resuming an **old** checkpoint without that field may print a **meta mismatch** warning (safe to continue or retrain from scratch).
