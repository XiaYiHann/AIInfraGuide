"""Minimal CPU tests for examples/pytorch/module_state.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import module_state  # noqa: E402


class ModuleStateTest(unittest.TestCase):
    def test_state_round_trip_uses_an_independent_snapshot(self) -> None:
        result = module_state.round_trip_demo()
        self.assertEqual(result["scale"], 3.0)
        self.assertEqual(result["running_total"], [1.0, 2.0, 3.0])
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["unexpected_keys"], [])

        shallow = module_state.shallow_reference_demo()
        self.assertTrue(shallow["same_storage"])
        self.assertEqual(shallow["before"], 2.0)
        self.assertEqual(shallow["after"], 5.0)

    def test_buffer_persistence_and_plain_attribute_boundaries(self) -> None:
        registration = module_state.registration_demo()
        self.assertIn("running_total", registration["buffer_names"])
        self.assertIn("scratch", registration["buffer_names"])
        self.assertIn("running_total", registration["state_keys"])
        self.assertNotIn("scratch", registration["state_keys"])
        self.assertNotIn("plain_tensor", registration["state_keys"])

        conversion = module_state.conversion_demo()
        self.assertEqual(conversion["parameter"], torch.float64)
        self.assertEqual(conversion["persistent_buffer"], torch.float64)
        self.assertEqual(conversion["temporary_buffer"], torch.float64)
        self.assertEqual(conversion["plain_tensor"], torch.float32)

        restored = module_state.round_trip_demo()
        self.assertEqual(restored["scratch"], [14.0, 14.0, 14.0])
        self.assertEqual(restored["plain_tensor"], [15.0, 15.0, 15.0])

    def test_registered_containers_avoid_python_list_failure(self) -> None:
        registration = module_state.registration_demo()
        self.assertIn("layers.0.weight", registration["parameter_names"])
        self.assertIn("scales.0", registration["parameter_names"])
        self.assertTrue(registration["python_module_list_missing"])
        self.assertTrue(registration["python_parameter_list_missing"])

        conversion = module_state.conversion_demo()
        self.assertEqual(conversion["python_list_module"], torch.float32)
        self.assertEqual(conversion["python_list_parameter"], torch.float32)

    def test_strict_load_rejects_and_non_strict_load_diagnoses(self) -> None:
        result = module_state.strict_load_demo()
        self.assertEqual(result["missing_keys"], ["projection.weight"])
        self.assertEqual(result["unexpected_keys"], ["legacy.weight"])
        self.assertIn("Missing key(s)", result["strict_error"])
        self.assertIn("Unexpected key(s)", result["strict_error"])

    def test_shared_parameter_identity_survives_state_loading(self) -> None:
        result = module_state.sharing_demo()
        self.assertTrue(result["same_object"])
        self.assertTrue(result["same_storage"])
        self.assertEqual(result["state_keys"], ["encoder_scale", "decoder_scale"])
        self.assertEqual(result["named_parameters"], ["encoder_scale"])
        self.assertEqual(result["restored_value"], [1.0, -1.0])

    def test_train_eval_initialization_and_module_hooks(self) -> None:
        model = module_state.ModuleStateDemo()
        model.apply(module_state.initialize_linears)
        for child in model.modules():
            if isinstance(child, torch.nn.Linear):
                torch.testing.assert_close(
                    child.weight,
                    torch.full_like(child.weight, 0.25),
                )

        result = module_state.mode_and_hook_demo()
        self.assertTrue(result["all_training_before"])
        self.assertTrue(result["all_eval_after"])
        self.assertTrue(result["output_requires_grad_in_eval"])
        self.assertTrue(result["parameter_still_requires_grad"])
        self.assertEqual(result["events"], ["forward_pre", "forward"])


if __name__ == "__main__":
    unittest.main()
