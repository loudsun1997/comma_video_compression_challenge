# Compression experiments log

**Objective:** minimize  
`score = 100 × segnet_dist + √(10 × posenet_dist) + 25 × rate`  
(**lower is better** — a higher number is a worse result.)

**Primary machine:** Apple Silicon Mac — use **`uv sync --group mps`**, **`evaluate.sh`** defaults to **`mps`** on macOS, and optional **`dev_videotoolbox_quick`** for fast encodes. Linux notes are at the end of the Mac workflow section.

**Evaluation (Mac quick path):** from repo root, after `uv sync --group mps`:
`bash evaluate.sh --quick --submission-dir ./submissions/<name>`  
(`--quick` → `tiny_test.txt`; on macOS, **MPS** is the default device unless you pass `--device cpu`.)

---

## Apple Silicon (Mac) workflow

1. **PyTorch / evaluation on GPU**  
   Install the MPS build: `uv sync --group mps` (not `--group cpu`).  
   On **macOS**, `evaluate.sh` defaults to **`--device mps`** unless you set `--device` explicitly.  
   For a fast loop, add **`--quick`** (uses repo-root **`tiny_test.txt`**; same list you should pass to `compress.sh`).

2. **Fast compression (VideoToolbox)**  
   Example submission: `submissions/dev_videotoolbox_quick/` — uses `hevc_videotoolbox` and `-q:v` instead of libx265/CRF. On the ffmpeg build tested here, **lower `-q:v` → smaller files**; raise it for quality. Hardware encode is great for iterating on scale/GOP ideas; for best **rate** at a fixed quality, compare against **libx265** before a final submission.

3. **Tiny compress/eval loop**  
   Repo root **`tiny_test.txt`** lists a subset of videos to encode (right now the public list is only `0.mkv`, so the file has one line). When you have more names in `public_test_video_names.txt`, copy **3–4 lines** into `tiny_test.txt` (unique basenames; no duplicate `foo.mkv` rows in one run). Then from repo root:
   ```bash
   bash submissions/dev_videotoolbox_quick/compress.sh --video-names-file tiny_test.txt
   bash evaluate.sh --quick --submission-dir ./submissions/dev_videotoolbox_quick
   ```
   **`--quick`** sets the video list to **`tiny_test.txt`** automatically (equivalent to `--video-names-file tiny_test.txt`). Use the **same** list for **compress** and **eval** so the inflated `.raw` set matches what `evaluate.py` scores.  
   Full public eval (no quick flag): omit `--quick` and use default `public_test_video_names.txt`, or pass `--video-names-file public_test_video_names.txt` explicitly.

**Linux / CI (reference only):** defaults stay **`cpu`** unless you pass **`--device cuda`**. Use `uv sync --group cu126` (or `cu128` / `cpu` per `pyproject.toml`). `dev_videotoolbox_quick` is mac-only (VideoToolbox); on Linux use **`libx265`** compress scripts.

---

## Best result so far

| Metric | Value |
|--------|--------|
| **Submission** | `p6_bicubic_c27_ft1` |
| **Settings** | Same as **p4_kitchen_sink** but **`flags=bicubic`** on downscale + **`frame-threads=1`** in x265 |
| **Score** | **2.77** |
| SegNet | 0.005784 |
| PoseNet | 0.192 |
| Rate | 0.03240 |
| Archive (zip) | ~1.22 MB |

**Same recipe, 64-clip holdout** (`full_test_video_names.txt` after **`test_videos.zip`** + remux): **score 3.25** — see **Phase 8 — full roster** below (still under the **3.50** rough robustness bar).

**Previous best:** `p4_kitchen_sink_40_c27_gop60` (Lanczos, `frame-threads=4`) at **2.78**.

**Footnote — the “75 MB denominator glitch” (invalid score ~2.37):** An older **`evaluate.py`** computed **original uncompressed size** by summing **every file under `videos/`** (`rglob`), **ignoring** `--video-names-file`. If **`videos/`** held **two** `.mkv` files but the roster and zip only covered **`0.mkv`**, the **rate** denominator doubled while **SegNet / PoseNet** (and zip size) still reflected **one** clip — producing an artificial **~2.37**. **That number is not a real improvement.** **`evaluate.py` now sums only the listed videos** so **rate** stays consistent with compress + NN eval. (Red flags that caught it: **identical** distortions to **8 decimals** and an **unchanged** zip size vs a “64-video” story.)

---

## Phase 2 (strategy sweep)

| ID | Submission dir | Encoder settings (summary) | SegNet | PoseNet | Rate | Score |
|----|----------------|----------------------------|--------|---------|------|-------|
| P2-G1 | `p2_g1_medium_c30_gop60` | medium, CRF 30, 45%, **GOP 60**, `bf=2`, scenecut 40 | 0.009077 | 0.594 | 0.01686 | **3.77** |
| P2-P1 | `p2_p1_slower_c30_gop1` | **slower**, CRF 30, 45%, GOP 1 | 0.008351 | 0.292 | 0.05772 | **3.99** |
| P2-S1 | `p2_s1_slower_c31_gop1` | slower, CRF **31**, 45%, GOP 1 | 0.009160 | 0.385 | 0.05068 | 4.15 |
| P2-S2 | `p2_s2_slower_c32_gop1` | slower, CRF **32**, 45%, GOP 1 | 0.010097 | 0.519 | 0.04476 | 4.41 |
| P2-S3 | `p2_s3_slower_c33_gop1` | slower, CRF **33**, 45%, GOP 1 | 0.011112 | 0.680 | 0.03985 | 4.72 |
| P2-D1 | `p2_d1_slower_c27_scale40` | slower, CRF **27**, **40%** scale, GOP 1 | 0.006878 | 0.146 | 0.06982 | **3.64** |

**Takeaways (same clip, `mps`):**

- **Temporal (P2-G1):** GOP 60 slashed **rate** (~1.7% of original) vs all-intra; PoseNet worsened but not enough to erase the gain — strong **score** improvement vs `exp1_preset_medium` (4.06).
- **slower @ CRF 30, GOP 1 (P2-P1):** Better score than E1 (4.06 → 3.99) with modest rate improvement vs E1’s slightly larger zip.
- **CRF cliff (P2-S1–S3):** From CRF 31→33, PoseNet climbs quickly; **31** is still ok (4.15); **32** nearly ties baseline (4.41).
- **Smaller scale + stronger quality (P2-D1):** 40% + CRF 27 + `slower` + all-intra — excellent PoseNet / SegNet; superseded for **score** by **P3-M1** once GOP 60 is added.

---

## Phase 3 (merge + PoseNet mitigations)

| ID | Submission dir | Encoder settings (summary) | SegNet | PoseNet | Rate | Score |
|----|----------------|----------------------------|--------|---------|------|-------|
| P3-M1 | `p2_merge_40_c27_gop60` | 40%, **slower CRF 27**, GOP 60, bf=2 | 0.006300 | 0.325 | 0.02523 | **3.06** |
| P3-SAO | `p2_g1_gop60_nosao` | P2-G1 + **`no-sao=1`** in x265 | 0.009060 | 0.478 | 0.01648 | **3.50** |
| P3-P0 | `p2_g1_gop60_bf0` | P2-G1 but **`bf=0`** (no B-frames) | 0.007996 | 0.438 | 0.02192 | **3.44** |

**Takeaways:** Merging **40% + CRF 27 + slower** with **GOP 60** beats both parents on this clip: much smaller **rate** than P2-D1 with **PoseNet** far below raw P2-G1. **`bf=0`** and **`no-sao=1`** both improve on baseline P2-G1 (**3.77** → **3.44** / **3.50**). Stacking **both** on P3-M1 is **Phase 4 — see below**.

---

## Phase 4 (meta-merge + GOP sweep)

| ID | Submission dir | Encoder settings (summary) | SegNet | PoseNet | Rate | Score |
|----|----------------|----------------------------|--------|---------|------|-------|
| P4-KS | `p4_kitchen_sink_40_c27_gop60` | P3-M1 + **`bf=0`** + **`no-sao=1`** | 0.005751 | 0.190 | 0.03302 | **2.78** |
| P4-G20 | `p4_merge_40_c27_gop20` | P3-M1 but **GOP 20**, `bf=2` | 0.006354 | 0.314 | 0.02655 | **3.07** |
| P4-G120 | `p4_merge_40_c27_gop120` | P3-M1 but **GOP 120**, `bf=2` | 0.006178 | 0.348 | 0.02468 | **3.10** |

**Takeaways:** **Kitchen sink** lands in the **2.7x** range: PoseNet **0.19** (hypothesis “0.2xx” met), SegNet drops, rate rises vs P3-M1 (no B-frames + SAO off cost bits) but **25×rate** + metric gains win overall. **GOP 60** remains best among **20 / 60 / 120** for this merge recipe on `0.mkv`.

---

## Phase 5 (sky / hood blackout) — executed, **not** a win

| ID | Submission dir | Encoder settings (summary) | SegNet | PoseNet | Rate | Score |
|----|----------------|----------------------------|--------|---------|------|-------|
| P5-B | `p5_blackout_30_10` | **drawbox** top **30%** + bottom **10%** black → then same vf/encode as **p4_kitchen_sink** | 0.007350 | 0.738 | 0.02667 | **4.12** |

**Why it backfired:** Official eval compares **every pixel** of reconstructed frames to the **full** original (`evaluate.py` / `0.mkv`). Blacking sky/hood **changes** those regions vs ground truth, so SegNet/PoseNet see huge disagreement there — **PoseNet ~0.74** vs **~0.19** without blackout. **Rate** did drop slightly (0.033 → 0.027) vs kitchen sink, but not enough to offset the metric penalty.

**Implication:** A “spend bits only on the road” trick only helps if metrics ignore or mask non-road regions (or originals are preprocessed the same way). For this challenge, **keep full-frame fidelity**.

---

## Phase 6 (micro-optimizations)

| ID | Submission dir | Change vs `p4_kitchen_sink` | SegNet | PoseNet | Rate | Score |
|----|----------------|----------------------------|--------|---------|------|-------|
| P6-B | `p6_bicubic_c27` | **bicubic** downscale only (`inflate` uses bicubic up) | 0.005708 | 0.203 | 0.03243 | **2.80** |
| P6-C28 | `p6_bicubic_c28` | bicubic + **CRF 28** | 0.006302 | 0.261 | 0.02582 | **2.89** |
| P6-FT1 | `p6_bicubic_c27_ft1` | bicubic + **`frame-threads=1`** | 0.005784 | 0.192 | 0.03240 | **2.77** |

**Takeaways (same clip, `mps`):**

- **Bicubic alone** at CRF 27 was slightly **worse** than Lanczos (**2.80** vs **2.78**) — ffmpeg swscale bicubic ≠ identical to PyTorch’s `F.interpolate`, so “match inflate” is heuristic.
- **CRF 28** on the bicubic chain **hurt** the score (**2.89**): rate gain did not pay for PoseNet/SegNet here.
- **`frame-threads=1`** with bicubic produced the new best **2.77** (tiny **rate** win vs p4 + slightly different NN metrics). **Phase 7** isolates Lanczos + `ft=1` — see below.

---

## Phase 7 (final polish)

| ID | Submission dir | Change (vs prior best chain) | SegNet | PoseNet | Rate | Score |
|----|----------------|------------------------------|--------|---------|------|-------|
| P7-LZ1 | `p7_lanczos_c27_ft1` | **p4_kitchen_sink** + **`frame-threads=1`** (Lanczos) | 0.005809 | 0.195 | 0.03306 | **2.80** |
| P7-G30 | `p7_bicubic_c27_gop30_ft1` | **p6** champion + **GOP 30** (`keyint=30`) | 0.005860 | 0.226 | 0.03350 | **2.93** |
| P7-VS | `p7_bicubic_c27_veryslow_ft1` | **p6** champion + **`preset veryslow`** | 0.005736 | 0.213 | 0.03159 | **2.82** |

**Procedure:** Lanczos + `ft=1` did **not** beat **2.77** (worse than **2.78** p4 with `ft=4`), so **GOP 30** and **veryslow** were branched from **`p6_bicubic_c27_ft1`**.

**Takeaways:** **Threading + scaler interact:** bicubic + single-thread wins here; Lanczos + single-thread regresses to **2.80**. **GOP 30** is worse than **60** on this recipe (**2.93**). **`veryslow`** did **not** beat **`slower`** on `0.mkv` (**2.82** vs **2.77** — still worth a check on a larger holdout). **Floor for this clip:** **`p6_bicubic_c27_ft1` @ 2.77**.

---

## Phase 8 (generalization / full batch)

**CPU / wall-clock:** `p6_bicubic_c27_ft1/compress.sh` keeps **`frame-threads=1`** inside x265 (one core per *file*). The script now defaults **`JOBS`** to **logical CPU count** (`sysctl -n hw.logicalcpu` on macOS, else `nproc`) so **`xargs -P$JOBS`** runs that many **parallel encodes** on distinct files. Override with e.g. `--jobs 4` if RAM thermals spike.

**Sample count in the report:** `evaluate.py` aggregates distortions over **all** sequences from the listed videos (weighted by batch). One public **`0.mkv`** run is often **~600** samples; **64** remuxed clips produced **38,380** samples in the logged Phase 8 run — the headline number scales with the roster, not “always 600.”

**Important — `test_videos.zip` is not a bag of `.mkv` files:** the archive contains **`.hevc`** bitstreams under **nested paths** (often each file is named `video.hevc`). **`unzip -j` into `videos/` is wrong** — every entry collides on the same basename and you end up with **one** overwritten file and **no** `.mkv` roster. Use **`download_and_remux.sh`** (segment list → **`0.mkv`**) or **`./tempCommand.sh`**, which unzips with paths preserved, **sorts** all **`*.hevc`**, and **remuxes** with **`ffmpeg`** to **`videos/0.mkv`, `1.mkv`, …** (same idea as **`download_and_remux.sh`**). Requires **`ffmpeg`** on **`PATH`**.

**Full run (no tiny harness):**

```bash
bash submissions/p6_bicubic_c27_ft1/compress.sh
bash evaluate.sh --submission-dir ./submissions/p6_bicubic_c27_ft1
```

Use **`public_test_video_names.txt`** (default) — not **`tiny_test.txt`**. On macOS, **`evaluate.sh`** defaults to **`mps`**.

**Verify `videos/` before a real multi-clip run:** `ls -lh videos/*.mkv | wc -l` should be **~64** after a successful remux. If you only see a lone **`video.hevc`** or **`0.mkv`**, you used a bad extract path — re-run **`./tempCommand.sh --redownload`** (or **`download_and_remux.sh`** for the small public segment list).

**Rebuild roster** (repo root — basenames only, no path junk):

```bash
ls -1 videos/ | grep '\.mkv$' > full_test_video_names.txt
wc -l full_test_video_names.txt
```

Expect **~64** lines when the full set is present; if **`wc -l`** prints **1**, stop and fix **`videos/`**.

**Same submission, custom roster** (compress and eval **must** use the **same** `--video-names-file`):

```bash
bash submissions/p6_bicubic_c27_ft1/compress.sh --video-names-file full_test_video_names.txt
bash evaluate.sh --submission-dir ./submissions/p6_bicubic_c27_ft1 --device mps --video-names-file full_test_video_names.txt
```

Or run **`./tempCommand.sh`** from the challenge root (download zip if needed → unzip **with paths** → remux **`.hevc` → `.mkv`** → **`full_test_video_names.txt`** → compress → eval). Use **`./tempCommand.sh --redownload`** after a failed flat unzip.

**Leaderboard reference:** `baseline_fast` is ~**4.4** on the public harness; **2.77** is on the default **`0.mkv`** list — **re-score on the full roster** once **`videos/`** and the list are aligned.

**Canonical report:** **`submissions/p6_bicubic_c27_ft1/report.txt`** (overwritten each eval — note which **`video_names_file`** you used).

**Phase 8 — single-clip public harness** (`public_test_video_names.txt`, **`0.mkv`**, **`device: mps`**):

| | Value |
|--|--------|
| Samples | 600 |
| SegNet | 0.00578428 |
| PoseNet | 0.19210856 |
| Rate | 0.03240365 |
| **Score** | **2.77** |
| Zip | 1,216,611 bytes |

**Phase 8 — full 64-clip roster** (`full_test_video_names.txt`, **`0.mkv`…`63.mkv`**, remux via **`./tempCommand.sh`**, **`device: mps`**):

| | Value |
|--|--------|
| Samples | 38,380 |
| SegNet | 0.00812918 |
| PoseNet | 0.19821380 |
| Rate | 0.04099735 |
| **Score** | **3.25** |
| Zip | 98,405,862 bytes |
| Sum of listed originals | 2,400,298,091 bytes |

(Update these tables after each run; paste from `report.txt`.)

---

## Earlier runs (full history)

| ID | Submission dir | Encoder settings (summary) | SegNet | PoseNet | Rate | Score |
|----|----------------|----------------------------|--------|---------|------|-------|
| B0 | `baseline_fast` | ultrafast, CRF 30, 45% Lanczos, GOP 1 | 0.009501 | 0.392 | 0.05984 | 4.43 |
| E1 | `exp1_preset_medium` | medium, CRF 30, 45%, GOP 1 | 0.007571 | 0.246 | 0.06914 | 4.06 |
| E2 | `exp2_scale55_crf35` | ultrafast, CRF 35, 55% | 0.012311 | 1.518 | 0.04025 | 6.13 |
| E3 | `exp3_scale_bicubic` | ultrafast, CRF 30, 45% bicubic downscale | 0.009481 | 0.421 | 0.05828 | 4.46 |
| P1 | `plan1_exp1_medium_crf34` | medium, CRF 34, 45%, GOP 1 | 0.011522 | 0.856 | 0.03946 | 5.07 |

---

## How to add a row

1. Create or edit `submissions/<name>/compress.sh`.
2. Run `bash submissions/<name>/compress.sh`.
3. Run `bash evaluate.sh --submission-dir ./submissions/<name> --device <device>`.
4. Copy SegNet, PoseNet, Rate, and Final score from `submissions/<name>/report.txt` into the table.
5. If the new **score** is lower than the current best, update the **Best result so far** section.

---

## Ideas not run / follow-ups

- **Learned upscaler (ML phase):** **`learned_upscaler/`** — **`TS_SPCN`** footprint + tests; log training / infra in **`learned_upscaler/ML_EXPERIMENTS.md`**. **Neural submission smoke:** **`submissions/p10_neural_infra/`** (25% scale + weights in zip + MPS/cuda inflate); see **`ML_EXPERIMENTS.md` Step 2** for timings.
- **Full-roster rescore:** **`p7_bicubic_c27_veryslow_ft1`** (or other variants) on **`full_test_video_names.txt`** — `veryslow` sometimes pays only at scale; **`p6`** logged at **3.25** on **64** clips.
- **GOP 45 / 75** around **60** on the champion chain if you want a finer GOP curve.
- **CRF 29** on **`p6`** if you need more **rate** without another structural change.
- **55% + slow + CRF 36** (earlier Plan 1 Exp 2) — still unlogged here.
