"""Fast CPU tests for examples/pytorch/training_loop.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import training_loop  # noqa: E402


class TrainingLoopTest(unittest.TestCase):
    def test_loss_decreases_on_fixed_cpu_data(self) -> None:
        result = training_loop.loss_decrease_demo()
        self.assertLess(result["after"], result["before"] * 0.2)

    def test_equal_sized_accumulation_matches_large_batch(self) -> None:
        difference = training_loop.accumulation_equivalence_demo()
        self.assertLess(difference, 1e-6)

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        unequal = [
            (torch.ones(2, 2), torch.ones(2, 1)),
            (torch.ones(1, 2), torch.ones(1, 1)),
        ]
        with self.assertRaisesRegex(ValueError, "equal-sized micro-batches"):
            training_loop.train_one_update(
                model,
                optimizer,
                None,
                unequal,
                training_loop.make_amp_policy("cpu", requested=False),
            )

    def test_gradient_clipping_bounds_global_norm(self) -> None:
        result = training_loop.clipping_demo(max_norm=0.2)
        self.assertGreater(result["pre_clip_norm"], result["max_norm"])
        self.assertLessEqual(
            result["post_clip_norm"], result["max_norm"] + 1e-6
        )

    def test_checkpoint_continues_the_exact_training_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            result = training_loop.exact_resume_demo(path)
            self.assertTrue(path.exists())
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tmp")), []
            )

        self.assertTrue(result["losses_match"])
        self.assertTrue(result["parameters_match"])
        self.assertTrue(result["learning_rate_match"])
        self.assertEqual(result["progress"]["global_step"], 3)
        self.assertEqual(
            result["progress"]["data_position"]["next_batch_index"], 6
        )

    def test_checkpoint_rejects_unsafe_resume_boundaries_and_bad_schema(self) -> None:
        model, optimizer, scheduler = training_loop.build_components(seed=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            with self.assertRaisesRegex(ValueError, "update boundaries"):
                training_loop.atomic_save_checkpoint(
                    path,
                    model,
                    optimizer,
                    scheduler,
                    None,
                    global_step=0,
                    data_position={
                        "epoch": 0,
                        "next_batch_index": 1,
                        "micro_step_in_update": 1,
                    },
                )
            self.assertFalse(path.exists())

            torch.save({"format_version": 1}, path)
            with self.assertRaisesRegex(ValueError, "missing keys"):
                training_loop.load_checkpoint(
                    path, model, optimizer, scheduler, None
                )

    def test_cpu_amp_request_degrades_explicitly_to_float32(self) -> None:
        policy = training_loop.make_amp_policy("cpu", requested=True)
        self.assertFalse(policy.enabled)
        self.assertEqual(policy.dtype, torch.float32)
        self.assertIsNone(policy.scaler)
        self.assertIn("CPU fallback", policy.reason)


if __name__ == "__main__":
    unittest.main()
