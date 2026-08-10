"""CPU fallback and CUDA correctness tests for cuda_async.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import cuda_async  # noqa: E402


CUDA_REQUIRED = unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA unavailable: stream, Event, H2D, and GPU timing checks are skipped",
)


class CudaAsyncTest(unittest.TestCase):
    def test_cpu_only_environment_degrades_explicitly(self) -> None:
        status = cuda_async.cpu_fallback_status(cuda_available=False)
        self.assertIn("CUDA unavailable", status)
        self.assertIn("skipped streams, Events, pinned H2D, and GPU timing", status)
        self.assertIn("performance was not measured", status)

    @CUDA_REQUIRED
    def test_current_stream_context_selects_side_and_restores_previous(self) -> None:
        before = torch.cuda.current_stream()
        observation = cuda_async.observe_stream_context()
        self.assertTrue(observation.context_selected_side_stream)
        self.assertTrue(observation.context_restored_previous_stream)
        self.assertEqual(torch.cuda.current_stream(), before)

    @CUDA_REQUIRED
    def test_event_wait_stream_and_record_stream_pipeline_is_correct(self) -> None:
        result = cuda_async.event_dependency_pipeline(length=257)
        expected = torch.arange(257, dtype=torch.int64).mul_(2).add_(1)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(result, expected))

        with self.assertRaisesRegex(ValueError, "length must be positive"):
            cuda_async.event_dependency_pipeline(length=0)

    @CUDA_REQUIRED
    def test_pinned_non_blocking_h2d_copy_is_correct(self) -> None:
        observation = cuda_async.non_blocking_h2d_pipeline(length=33)
        self.assertTrue(observation.source_was_pinned)
        self.assertTrue(observation.copy_used_non_default_stream)
        self.assertEqual(observation.copied_values, list(range(33)))

    @CUDA_REQUIRED
    def test_warmup_and_event_timing_have_explicit_completion_boundaries(self) -> None:
        observation = cuda_async.observe_enqueue_and_timing(
            elements=1 << 16,
            warmup_iterations=2,
            measured_iterations=3,
        )
        self.assertEqual(observation.warmup_iterations, 2)
        self.assertEqual(observation.measured_iterations, 3)
        self.assertGreaterEqual(observation.enqueue_seconds, 0.0)
        self.assertGreaterEqual(observation.event_milliseconds, 0.0)
        self.assertGreaterEqual(observation.synchronized_wall_seconds, 0.0)
        # A fast device may already have completed at query time, so this is
        # deliberately a type check rather than an assertion that it is pending.
        self.assertIsInstance(
            observation.completion_was_pending_after_enqueue, bool
        )

        with self.assertRaisesRegex(ValueError, "measured_iterations"):
            cuda_async.observe_enqueue_and_timing(measured_iterations=0)


if __name__ == "__main__":
    unittest.main()
