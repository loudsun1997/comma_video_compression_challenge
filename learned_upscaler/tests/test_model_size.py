"""Sanity checks for TS-SPCN size budget and I/O shapes."""
import os
import tempfile
import unittest

import torch

from learned_upscaler.model import TS_SPCN


class TestTSSPCN(unittest.TestCase):
    def test_forward_4x_shape(self):
        m = TS_SPCN(upscale_factor=4)
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 3, 128, 128))

    def test_state_dict_under_100kb(self):
        m = TS_SPCN(upscale_factor=4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(m.state_dict(), path)
            kb = os.path.getsize(path) / 1024
            self.assertLess(
                kb,
                100.0,
                f"state_dict checkpoint should stay under 100 KB for rate budget (got {kb:.2f} KB)",
            )
        finally:
            os.unlink(path)

    def test_parameter_count_stable(self):
        m = TS_SPCN(upscale_factor=4, num_residual_blocks=4)
        n = sum(p.numel() for p in m.parameters())
        self.assertEqual(n, 18_520)


if __name__ == "__main__":
    unittest.main()
