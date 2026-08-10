"""CPU ledger and optional CUDA correctness tests for memory_accounting.py."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import memory_accounting  # noqa: E402


CUDA_REQUIRED = unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA unavailable: peak reset and allocated/reserved checks are skipped",
)
_CUDA_OBSERVATION: memory_accounting.CudaMemoryObservation | None = None


def cuda_observation() -> memory_accounting.CudaMemoryObservation:
    global _CUDA_OBSERVATION
    if _CUDA_OBSERVATION is None:
        _CUDA_OBSERVATION = memory_accounting.run_cuda_memory_smoke()
    return _CUDA_OBSERVATION


class MemoryAccountingCpuTest(unittest.TestCase):
    def test_json_ledger_parses_and_aggregates_exact_bytes(self) -> None:
        payload = json.dumps(
            [
                {
                    "category": "parameters",
                    "name": "weight",
                    "numel": 8,
                    "bytes_per_element": 4,
                },
                {
                    "category": "gradients",
                    "name": "weight.grad",
                    "numel": 8,
                    "bytes_per_element": 4,
                },
                {
                    "category": "activations",
                    "name": "hidden",
                    "numel": 6,
                    "bytes_per_element": 2,
                },
            ]
        )
        entries = memory_accounting.parse_ledger_json(payload)
        totals = memory_accounting.aggregate_ledger(entries)

        self.assertEqual(totals["parameters"], 32)
        self.assertEqual(totals["gradients"], 32)
        self.assertEqual(totals["activations"], 12)
        self.assertEqual(totals["optimizer_state"], 0)
        self.assertEqual(totals["temporary_buffers"], 0)
        self.assertEqual(memory_accounting.total_ledger_bytes(entries), 76)

    def test_invalid_ledger_records_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown memory category"):
            memory_accounting.parse_ledger_records(
                [
                    {
                        "category": "mystery",
                        "name": "x",
                        "numel": 1,
                        "bytes_per_element": 4,
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "numel must be non-negative"):
            memory_accounting.LedgerEntry("parameters", "x", -1, 4)
        with self.assertRaisesRegex(ValueError, "missing=.*bytes_per_element"):
            memory_accounting.parse_ledger_records(
                [{"category": "parameters", "name": "x", "numel": 1}]
            )
        with self.assertRaisesRegex(TypeError, "must contain a list"):
            memory_accounting.parse_ledger_json('{"category": "parameters"}')

    def test_cpu_demo_materializes_all_five_categories_and_round_trips(self) -> None:
        entries = memory_accounting.cpu_ledger_demo()
        totals = memory_accounting.aggregate_ledger(entries)

        self.assertEqual(set(totals), set(memory_accounting.MEMORY_CATEGORIES))
        for category in memory_accounting.MEMORY_CATEGORIES:
            with self.subTest(category=category):
                self.assertGreater(totals[category], 0)
        self.assertEqual(sum(totals.values()), memory_accounting.total_ledger_bytes(entries))

    def test_tensor_entry_uses_numel_times_element_size(self) -> None:
        tensor = torch.zeros(3, 5, dtype=torch.float64)
        entry = memory_accounting.tensor_entry("temporary_buffers", "scratch", tensor)
        self.assertEqual(entry.numel, 15)
        self.assertEqual(entry.bytes_per_element, 8)
        self.assertEqual(entry.nbytes, 120)

    def test_no_cuda_message_names_unverified_checks(self) -> None:
        status = memory_accounting.cuda_fallback_status(cuda_available=False)
        self.assertIn("CUDA unavailable", status)
        self.assertIn("peak reset", status)
        self.assertIn("allocated/reserved", status)
        self.assertIn("performance were not measured", status)


class MemoryAccountingCudaTest(unittest.TestCase):
    @CUDA_REQUIRED
    def test_small_cuda_training_smoke_is_correct(self) -> None:
        observation = cuda_observation()
        self.assertTrue(observation.loss_was_finite)
        self.assertTrue(observation.parameter_changed)
        self.assertEqual(
            [point.label for point in observation.points],
            [
                "after_peak_reset",
                "after_model",
                "after_forward",
                "after_backward",
                "after_optimizer_step",
            ],
        )

    @CUDA_REQUIRED
    def test_peak_reset_and_allocated_reserved_relations(self) -> None:
        observation = cuda_observation()
        self.assertEqual(
            observation.peak_immediately_after_reset,
            observation.allocated_at_peak_reset,
        )
        for point in observation.points:
            with self.subTest(point=point.label):
                self.assertLessEqual(point.allocated, point.reserved)
        self.assertGreaterEqual(
            observation.peak_allocated,
            max(point.allocated for point in observation.points),
        )
        self.assertGreaterEqual(
            observation.peak_reserved,
            max(point.reserved for point in observation.points),
        )


if __name__ == "__main__":
    unittest.main()
