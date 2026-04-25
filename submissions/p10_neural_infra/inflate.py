#!/usr/bin/env python
import os
import sys

import av
import torch
import torch.nn.functional as F
from tqdm import tqdm

from frame_utils import camera_size, yuv420_to_rgb

from .model import TS_SPCN

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

_model = TS_SPCN(upscale_factor=4).to(device)
_model_path = os.path.join(os.path.dirname(__file__), "micro_upscaler.pt")
if os.path.isfile(_model_path):
    try:
        sd = torch.load(_model_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(_model_path, map_location=device)
    _model.load_state_dict(sd)
_model.eval()


@torch.inference_mode()
def decode_and_resize_to_file(video_path: str, dst: str) -> int:
    target_w, target_h = camera_size
    fmt = "hevc" if video_path.endswith(".hevc") else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    fr = getattr(stream, "frames", None)
    nb = int(fr) if fr is not None and fr > 0 else None
    n = 0
    with open(dst, "wb") as f:
        for frame in tqdm(
            container.decode(stream),
            desc=f"Neural inflate {os.path.basename(video_path)}",
            total=nb,
            unit="fr",
            leave=False,
        ):
            t = yuv420_to_rgb(frame)
            h, w, _ = t.shape
            if h != target_h or w != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                sr = _model(x)
                if sr.shape[2] != target_h or sr.shape[3] != target_w:
                    sr = F.interpolate(
                        sr, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                t = (sr.clamp(0, 1) * 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8).cpu()
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize_to_file(src, dst)
    print(f"saved {n} frames")
