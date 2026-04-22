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

        # 1. Semantic Feature Extractor (Wider receptive field to find proto-edges)
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.PReLU(32),
        )

        # 2. The Bottleneck (Shrink to 12 channels to save FLOPs and File Size)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 12, kernel_size=1),
            nn.PReLU(12),
        )

        # 3. Residual Mapping Stack (The "Brain" of the network)
        # Keeps math in the low-res space. 12 channels keeps parameters tiny.
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(12))
        self.mapping = nn.Sequential(*res_blocks)

        # 4. Semantic Upsampler (Prepare channels for PixelShuffle)
        # 3 output channels (RGB) * upscale_factor^2
        out_channels = 3 * (upscale_factor**2)
        self.upsample_prep = nn.Conv2d(12, out_channels, kernel_size=3, padding=1)

        # The magic layer: Reshapes [B, 48, H, W] -> [B, 3, H*4, W*4] with zero parameters
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

    # Instantiate the model for a 4x upscale (meaning we compress at 25% scale)
    model = TS_SPCN(upscale_factor=4)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # Save the model weights to disk to check the physical file size
    ckpt_path = _root / "micro_upscaler.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Check file size
    file_size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"Physical File Size: {file_size_kb:.2f} KB")
