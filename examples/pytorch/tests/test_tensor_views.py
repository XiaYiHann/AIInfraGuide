"""Minimal CPU tests for examples/pytorch/tensor_views.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import tensor_views  # noqa: E402


class TensorViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tensors = tensor_views.make_examples()

    def test_view_shares_storage_and_preserves_values(self) -> None:
        base = self.tensors["base"]
        viewed = self.tensors["viewed"]

        self.assertTrue(tensor_views.shares_storage(base, viewed))
        viewed[0, 0] = 99
        self.assertEqual(base[0, 0].item(), 99)

    def test_transpose_and_permute_only_change_metadata(self) -> None:
        base = self.tensors["base"]
        transposed = self.tensors["transposed"]
        permuted = self.tensors["permuted"]

        self.assertEqual(transposed.stride(), (1, 4))
        self.assertFalse(transposed.is_contiguous())
        self.assertTrue(tensor_views.shares_storage(base, transposed))
        self.assertTrue(tensor_views.shares_storage(base, permuted))
        self.assertTrue(torch.equal(transposed, base.t()))

    def test_view_rejects_incompatible_layout_while_reshape_copies(self) -> None:
        base = self.tensors["base"]
        transposed = self.tensors["transposed"]
        reshaped = self.tensors["reshaped"]

        with self.assertRaises(RuntimeError):
            transposed.view(-1)
        self.assertTrue(reshaped.is_contiguous())
        self.assertFalse(tensor_views.shares_storage(base, reshaped))
        self.assertTrue(torch.equal(reshaped, transposed.flatten()))

    def test_contiguous_copies_only_when_needed(self) -> None:
        base = self.tensors["base"]
        transposed = self.tensors["transposed"]
        contiguous = self.tensors["contiguous"]

        self.assertIs(base.contiguous(), base)
        self.assertTrue(contiguous.is_contiguous())
        self.assertFalse(tensor_views.shares_storage(transposed, contiguous))
        self.assertTrue(torch.equal(transposed, contiguous))

    def test_broadcast_and_overlapping_view_alias_storage(self) -> None:
        broadcast = self.tensors["broadcast"]
        overlap = self.tensors["overlap"]

        self.assertEqual(broadcast.stride(), (1, 0))
        self.assertTrue(
            tensor_views.shares_storage(self.tensors["broadcast_source"], broadcast)
        )
        self.assertEqual(
            tensor_views.storage_indices(overlap),
            [0, 1, 1, 2, 2, 3],
        )
        # 不对重叠 as_strided 视图执行原地写：官方文档将其行为定义为未定义。
        self.assertLess(
            len(set(tensor_views.storage_indices(overlap))),
            overlap.numel(),
        )


if __name__ == "__main__":
    unittest.main()
