import os
from pathlib import Path

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class TS_SPCN(nn.Module):
    def __init__(self, upscale_factor=4, num_residual_blocks=4):
        super(TS_SPCN, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.PReLU(32),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 12, kernel_size=1),
            nn.PReLU(12),
        )

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(12))
        self.mapping = nn.Sequential(*res_blocks)

        out_channels = 3 * (upscale_factor**2)
        self.upsample_prep = nn.Conv2d(12, out_channels, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.extractor(x)
        x = self.bottleneck(x)
        x = self.mapping(x)
        x = self.upsample_prep(x)
        x = self.pixel_shuffle(x)
        return x


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent
    model = TS_SPCN(upscale_factor=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    ckpt_path = _root / "micro_upscaler.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Physical File Size: {os.path.getsize(ckpt_path) / 1024:.2f} KB")
