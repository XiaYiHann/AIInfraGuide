"""Minimal CPU tests for examples/pytorch/autograd_mechanics.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import autograd_mechanics  # noqa: E402


class AutogradMechanicsTest(unittest.TestCase):
    def test_dynamic_graph_and_leaf_semantics(self) -> None:
        dynamic = autograd_mechanics.dynamic_graph_demo()
        self.assertNotEqual(dynamic["square_grad_fn"], dynamic["linear_grad_fn"])
        self.assertEqual(dynamic["square_gradient"], 4.0)
        self.assertEqual(dynamic["linear_gradient"], 3.0)

        leaf = autograd_mechanics.leaf_and_saved_tensor_demo()
        self.assertTrue(leaf["leaf_is_leaf"])
        self.assertTrue(leaf["leaf_grad_fn_is_none"])
        self.assertFalse(leaf["non_leaf_is_leaf"])
        self.assertEqual(leaf["leaf_gradient"], 36.0)
        self.assertEqual(leaf["retained_non_leaf_gradient"], 12.0)
        self.assertEqual(leaf["saved_tensor_count"], 1)
        self.assertTrue(leaf["saved_tensor_matches_input"])

    def test_backward_accumulates_into_leaf_grad(self) -> None:
        self.assertEqual(
            autograd_mechanics.gradient_accumulation_demo(),
            (3.0, 7.0),
        )

    def test_retain_graph_and_create_graph_have_distinct_jobs(self) -> None:
        result = autograd_mechanics.graph_control_demo()
        self.assertEqual(result["first_backward"], 6.0)
        self.assertEqual(result["second_backward"], 6.0)
        self.assertEqual(result["first_derivative"], 12.0)
        self.assertEqual(result["second_derivative"], 12.0)

    def test_detach_no_grad_and_inference_mode_boundaries(self) -> None:
        result = autograd_mechanics.mode_demo()
        self.assertFalse(result["detached_requires_grad"])
        self.assertTrue(result["detached_shares_storage"])
        self.assertFalse(result["no_grad_requires_grad"])
        self.assertEqual(result["no_grad_reuse_gradient"], [3.0, 6.0])
        self.assertTrue(result["inference_reuse_failed"])
        self.assertIn("Inference tensors", result["inference_error"])
        self.assertEqual(result["clone_reuse_gradient"], [1.0, 1.0])

    def test_tensor_hook_observes_and_replaces_gradient(self) -> None:
        result = autograd_mechanics.hook_demo()
        self.assertEqual(result["hook_observed"], [4.0])
        self.assertEqual(result["leaf_gradient_after_hook"], 12.0)

    def test_inplace_mutation_triggers_version_counter_error(self) -> None:
        result = autograd_mechanics.inplace_version_error()
        self.assertEqual(result["version_after"], result["version_before"] + 1)
        self.assertIn("modified by an inplace operation", result["error"].lower())

    def test_custom_function_passes_gradcheck(self) -> None:
        value = torch.tensor(
            [-0.7, 0.2, 1.1],
            dtype=torch.double,
            requires_grad=True,
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                autograd_mechanics.CustomSinh.apply,
                (value,),
            )
        )

    def test_gradcheck_rejects_wrong_backward(self) -> None:
        value = torch.tensor(
            [-0.8, 0.4, 1.2],
            dtype=torch.double,
            requires_grad=True,
        )
        with self.assertRaises(RuntimeError):
            torch.autograd.gradcheck(
                autograd_mechanics.WrongSquare.apply,
                (value,),
            )


if __name__ == "__main__":
    unittest.main()
