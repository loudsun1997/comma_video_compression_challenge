#!/usr/bin/env python
"""
Compression-aware overfit: ffmpeg+x265 low-res (match p10), then train TS-SPCN toward
full-res targets. Default loss: VGG16 relu2_2 perceptual + L1 anchor; use --loss l1
for pixel-only baseline.

Temp LR (--temp-lr): by default, reuse existing file if it is not older than --video
(mtime); otherwise encode. --force-encode always re-encodes; --skip-encode never does.

Step 6: optional temporal consistency L1 on frame deltas (|SR_t-SR_{t-1}| vs |HR_t-HR_{t-1}|);
default --temporal-weight 0.5. Set to 0 for spatial-only (Step 5-style).
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import av
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Repo root (parent of learned_upscaler/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb  # noqa: E402
from learned_upscaler.model import TS_SPCN  # noqa: E402


class VGGPerceptualLoss(nn.Module):
    """L1 pixel anchor + weighted MSE on frozen VGG16 features (through relu2_2)."""

    def __init__(self, device: torch.device, *, feature_weight: float = 0.1):
        super().__init__()
        w = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=w).features[:9].to(device)
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()
        self.feature_weight = feature_weight
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        pixel_loss = self.l1(sr, hr)
        sr_v = (sr - self.mean) / self.std
        hr_v = (hr - self.mean) / self.std
        feat_loss = F.mse_loss(self.vgg(sr_v), self.vgg(hr_v))
        return pixel_loss + self.feature_weight * feat_loss


def ffmpeg_compressed_lr(hr_video: Path, lr_out: Path, scale: float) -> None:
    """Match p10 compress.sh: 25% bicubic + libx265 slower CRF27 GOP60, etc."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-r",
        "20",
        "-fflags",
        "+genpts",
        "-i",
        str(hr_video.resolve()),
        "-vf",
        f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=bicubic",
        "-c:v",
        "libx265",
        "-preset",
        "slower",
        "-crf",
        "27",
        "-g",
        "60",
        "-bf",
        "0",
        "-x265-params",
        "keyint=60:min-keyint=1:scenecut=40:no-sao=1:frame-threads=1:log-level=warning",
        "-r",
        "20",
        str(lr_out.resolve()),
    ]
    subprocess.run(cmd, check=True)


def load_video_uint8_stack(path: Path, *, desc: str, show_progress: bool) -> tuple[torch.Tensor, int, int]:
    """(N, 3, H, W) uint8 on CPU; decode matches inflate (yuv420_to_rgb)."""
    fmt = "hevc" if str(path).endswith(".hevc") else None
    container = av.open(str(path), format=fmt)
    stream = container.streams.video[0]
    fr = getattr(stream, "frames", None)
    total = int(fr) if fr is not None and fr > 0 else None
    frames: list[torch.Tensor] = []
    dec = container.decode(stream)
    if show_progress:
        dec = tqdm(dec, desc=desc, total=total, unit="fr", leave=False)
    for frame in dec:
        t = yuv420_to_rgb(frame)
        frames.append(t.permute(2, 0, 1))
    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    vid = torch.stack(frames, dim=0).contiguous()
    _, _, h, w = vid.shape
    return vid, h, w


class CompressionAwareFrameDataset(Dataset):
    """Aligned (lr_t, hr_t, lr_prev, hr_prev) uint8; prev is frame idx-1, or duplicate at 0."""

    def __init__(self, lr_u8: torch.Tensor, hr_u8: torch.Tensor):
        assert lr_u8.shape[0] == hr_u8.shape[0], (lr_u8.shape, hr_u8.shape)
        self.lr = lr_u8
        self.hr = hr_u8

    def __len__(self) -> int:
        return self.lr.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_idx = max(0, idx - 1)
        return self.lr[idx], self.hr[idx], self.lr[prev_idx], self.hr[prev_idx]


def _checkpoint_meta(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "loss": args.loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "vgg_feature_weight": args.vgg_feature_weight,
        "scale": args.scale,
        "temporal_weight": args.temporal_weight,
    }


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    next_epoch: int,
    args: argparse.Namespace,
) -> None:
    """next_epoch = 1-based index of the epoch to run next (repeat current if interrupted mid-epoch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "meta": _checkpoint_meta(args),
        "rng_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    if not path.is_file():
        raise SystemExit(f"--resume: missing checkpoint {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    meta = ckpt.get("meta") or {}
    cur = _checkpoint_meta(args)
    for k in cur:
        if meta.get(k) != cur[k]:
            print(
                f"WARNING: checkpoint meta mismatch {k!r}: ckpt={meta.get(k)!r} current={cur[k]!r}",
                flush=True,
            )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_cpu"].cpu())
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    return int(ckpt["next_epoch"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compression-aware train for TS-SPCN (L1 and/or VGG perceptual).")
    ap.add_argument("--video", type=Path, default=ROOT / "videos" / "0.mkv")
    ap.add_argument("--scale", type=float, default=0.25, help="linear scale factor (match compress.sh)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--temp-lr",
        type=Path,
        default=Path(__file__).resolve().parent / "temp_lr_train.mkv",
        help="ffmpeg low-res output path",
    )
    ap.add_argument("--device", type=str, default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "micro_upscaler.pt")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument(
        "--keep-temp-lr",
        action="store_true",
        help="do not delete temp low-res mkv after training",
    )
    ap.add_argument(
        "--skip-encode",
        action="store_true",
        help="never run ffmpeg: require existing --temp-lr (error if missing)",
    )
    ap.add_argument(
        "--force-encode",
        action="store_true",
        help="always run ffmpeg (overwrite --temp-lr), even if a fresh temp already exists",
    )
    ap.add_argument(
        "--loss",
        choices=("vgg", "l1"),
        default="vgg",
        help="vgg = L1 + VGG16 relu2_2 perceptual (default); l1 = pixel only",
    )
    ap.add_argument(
        "--vgg-feature-weight",
        type=float,
        default=0.1,
        help="multiplier on VGG feature MSE (only if --loss vgg)",
    )
    ap.add_argument(
        "--temporal-weight",
        type=float,
        default=0.5,
        help="L1 on (SR_t-SR_{t-1}) vs (HR_t-HR_{t-1}); 0 disables temporal term (spatial-only)",
    )
    ckpt_default = Path(__file__).resolve().parent / "checkpoints" / "train_resume.pt"
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ckpt_default,
        help="path for pause/resume state (model + optimizer + next epoch)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="load --checkpoint and continue from saved next_epoch",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="save train_resume.pt after every N completed epochs (default 1 = survives hard shutdown after each epoch; 0 = off)",
    )
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tw, th = camera_size
    prog = not args.no_progress

    def log(msg: str) -> None:
        print(msg, flush=True)

    if not args.video.is_file():
        raise SystemExit(f"Missing video: {args.video}")

    log(f"Device: {device}")
    if args.force_encode and args.skip_encode:
        raise SystemExit("Use only one of --force-encode and --skip-encode")

    run_ffmpeg: bool
    if args.force_encode:
        run_ffmpeg = True
    elif args.skip_encode:
        if not args.temp_lr.is_file():
            raise SystemExit(f"--skip-encode but missing {args.temp_lr}")
        run_ffmpeg = False
        log(f"Using existing low-res file (--skip-encode): {args.temp_lr}")
    elif args.temp_lr.is_file():
        v_mtime = args.video.stat().st_mtime
        t_mtime = args.temp_lr.stat().st_mtime
        if v_mtime <= t_mtime:
            run_ffmpeg = False
            log(
                f"Auto: reusing {args.temp_lr} (source video not newer than temp). "
                f"Use --force-encode to rebuild."
            )
        else:
            run_ffmpeg = True
            log(f"Auto: source video newer than temp LR → re-encoding …")
    else:
        run_ffmpeg = True

    if run_ffmpeg:
        log(f"ffmpeg low-res → {args.temp_lr} (scale={args.scale}, match p10 x265) …")
        ffmpeg_compressed_lr(args.video, args.temp_lr, args.scale)
        log("ffmpeg done.")

    log("Loading high-res targets …")
    hr_u8, h, w = load_video_uint8_stack(
        args.video, desc="Load HR", show_progress=prog
    )
    if h != th or w != tw:
        log(f"Resizing HR {w}x{h} → {tw}x{th} (camera_size)")
        hr_u8 = (
            F.interpolate(hr_u8.float(), size=(th, tw), mode="bicubic", align_corners=False)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )

    log("Loading compressed low-res inputs …")
    lr_u8, _, _ = load_video_uint8_stack(
        args.temp_lr, desc="Load LR (HEVC)", show_progress=prog
    )

    n = min(hr_u8.shape[0], lr_u8.shape[0])
    if hr_u8.shape[0] != lr_u8.shape[0]:
        log(f"Truncating to min length {n} (HR {hr_u8.shape[0]}, LR {lr_u8.shape[0]})")
    hr_u8 = hr_u8[:n]
    lr_u8 = lr_u8[:n]

    dataset = CompressionAwareFrameDataset(lr_u8, hr_u8)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TS_SPCN(upscale_factor=4).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    if args.loss == "vgg":
        log(f"Loss: VGG perceptual (feature_weight={args.vgg_feature_weight}) + L1 anchor")
        crit = VGGPerceptualLoss(device, feature_weight=args.vgg_feature_weight)
    else:
        log("Loss: L1 only")
        crit = nn.L1Loss()
    if args.temporal_weight > 0:
        log(f"Temporal: L1(SR_delta vs HR_delta) weight={args.temporal_weight}")
    else:
        log("Temporal: off (--temporal-weight 0)")

    start_epoch = 1
    if args.resume:
        start_epoch = load_training_checkpoint(
            args.checkpoint, model=model, optimizer=opt, args=args, device=device
        )
        log(f"Resumed from {args.checkpoint}; starting at epoch {start_epoch}/{args.epochs}")

    _graceful_stop: dict[str, bool] = {"request": False}

    def _on_shutdown_signal(signum: int, frame: Any) -> None:
        _graceful_stop["request"] = True

    signal.signal(signal.SIGTERM, _on_shutdown_signal)
    try:
        signal.signal(signal.SIGHUP, _on_shutdown_signal)
    except (AttributeError, ValueError):
        pass

    training_finished = False
    try:
        epoch_range = range(start_epoch, args.epochs + 1)
        if prog:
            epoch_range = tqdm(
                epoch_range,
                desc="Epochs",
                unit="ep",
                total=max(args.epochs - start_epoch + 1, 0),
            )

        for epoch in epoch_range:
            model.train()
            running = 0.0
            batch_iter = loader
            if prog:
                batch_iter = tqdm(loader, desc=f"Train {epoch}/{args.epochs}", leave=False, unit="batch")
            try:
                for lr_t, hr_t, lr_prev, hr_prev in batch_iter:
                    if _graceful_stop["request"]:
                        save_training_checkpoint(
                            args.checkpoint,
                            model=model,
                            optimizer=opt,
                            next_epoch=epoch,
                            args=args,
                        )
                        log(
                            f"Shutdown signal: saved {args.checkpoint} (restart epoch {epoch}). "
                            f"Continue with: --resume"
                        )
                        raise SystemExit(143) from None
                    lr_t = lr_t.to(device).float() / 255.0
                    hr_t = hr_t.to(device).float() / 255.0
                    lr_prev = lr_prev.to(device).float() / 255.0
                    hr_prev = hr_prev.to(device).float() / 255.0

                    opt.zero_grad(set_to_none=True)
                    if args.temporal_weight > 0:
                        b = lr_t.size(0)
                        lr_combined = torch.cat([lr_t, lr_prev], dim=0)
                        sr_combined = model(lr_combined)
                        sr_t, sr_prev = sr_combined.split(b, dim=0)
                        if sr_t.shape[2] != hr_t.shape[2] or sr_t.shape[3] != hr_t.shape[3]:
                            sr_t = F.interpolate(
                                sr_t,
                                size=(hr_t.shape[2], hr_t.shape[3]),
                                mode="bilinear",
                                align_corners=False,
                            )
                        if sr_prev.shape[2] != hr_prev.shape[2] or sr_prev.shape[3] != hr_prev.shape[3]:
                            sr_prev = F.interpolate(
                                sr_prev,
                                size=(hr_prev.shape[2], hr_prev.shape[3]),
                                mode="bilinear",
                                align_corners=False,
                            )
                        spatial_loss = crit(sr_t, hr_t)
                        temporal_loss = F.l1_loss(sr_t - sr_prev, hr_t - hr_prev)
                        loss = spatial_loss + args.temporal_weight * temporal_loss
                    else:
                        sr = model(lr_t)
                        if sr.shape[2] != hr_t.shape[2] or sr.shape[3] != hr_t.shape[3]:
                            sr = F.interpolate(
                                sr,
                                size=(hr_t.shape[2], hr_t.shape[3]),
                                mode="bilinear",
                                align_corners=False,
                            )
                        loss = crit(sr, hr_t)
                    loss.backward()
                    opt.step()
                    running += float(loss.detach().cpu())
                    if prog and hasattr(batch_iter, "set_postfix"):
                        batch_iter.set_postfix(loss=f"{loss.detach().item():.5f}")
            except KeyboardInterrupt:
                save_training_checkpoint(
                    args.checkpoint,
                    model=model,
                    optimizer=opt,
                    next_epoch=epoch,
                    args=args,
                )
                log(
                    f"Paused: saved {args.checkpoint} (will restart epoch {epoch}). "
                    f"Continue with: --resume (same other flags; temp LR auto-reused if still valid)."
                )
                raise SystemExit(130) from None

            avg = running / max(len(loader), 1)
            if prog and hasattr(epoch_range, "set_postfix"):
                epoch_range.set_postfix(loss_avg=f"{avg:.5f}")
            elif args.no_progress and (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs):
                log(f"epoch {epoch:3d}/{args.epochs}  loss {avg:.6f}")

            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_training_checkpoint(
                    args.checkpoint,
                    model=model,
                    optimizer=opt,
                    next_epoch=epoch + 1,
                    args=args,
                )
                log(f"Checkpoint: {args.checkpoint} (next epoch {epoch + 1})")

        training_finished = True
    finally:
        if training_finished and (not args.keep_temp_lr) and args.temp_lr.is_file() and run_ffmpeg:
            args.temp_lr.unlink(missing_ok=True)
            log(f"Removed {args.temp_lr}")

    model.eval()
    torch.save(model.state_dict(), args.out)
    log(f"Saved {args.out} ({args.out.stat().st_size / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
