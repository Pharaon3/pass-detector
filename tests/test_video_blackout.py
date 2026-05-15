"""Temporal half-blackout on normalized clips."""

from __future__ import annotations

import unittest

import torch

from utils.video import apply_temporal_blackout_last_half, imagenet_black_normalized


class TestTemporalBlackout(unittest.TestCase):
    def test_second_half_is_black_normalized(self) -> None:
        t = torch.randn(1, 10, 3, 4, 5)
        out = apply_temporal_blackout_last_half(t)
        blk = imagenet_black_normalized(out.device, out.dtype).view(1, 1, 3, 1, 1)
        self.assertTrue(torch.allclose(out[:, 5:, ...], blk.expand(1, 5, 3, 4, 5)))
        self.assertTrue(torch.equal(out[:, :5, ...], t[:, :5, ...]))

    def test_odd_length_cut_point(self) -> None:
        t = torch.randn(1, 7, 3, 2, 2)
        out = apply_temporal_blackout_last_half(t)
        cut = 7 // 2  # 3
        blk = imagenet_black_normalized(out.device, out.dtype).view(1, 1, 3, 1, 1)
        self.assertTrue(torch.allclose(out[:, cut:, ...], blk.expand(1, 7 - cut, 3, 2, 2)))


if __name__ == "__main__":
    unittest.main()
