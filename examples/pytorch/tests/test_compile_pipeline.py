"""CPU tests for examples/pytorch/compile_pipeline.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import compile_pipeline  # noqa: E402


class CompilePipelineTest(unittest.TestCase):
    def test_eager_and_compiled_values_and_gradients_match(self) -> None:
        result = compile_pipeline.compare_eager_and_compile()
        self.assertEqual(result["backend"], "eager")
        self.assertEqual(result["output_shape"], [3])
        self.assertLessEqual(result["max_output_abs_error"], 1e-6)
        self.assertEqual(len(result["max_gradient_abs_errors"]), 3)
        self.assertTrue(
            all(error <= 1e-6 for error in result["max_gradient_abs_errors"])
        )

    def test_explain_reports_graphs_guards_and_no_break_for_normal_path(self) -> None:
        summary = compile_pipeline.explain_summary()
        self.assertGreaterEqual(summary["graph_count"], 1)
        self.assertEqual(summary["graph_break_count"], 0)
        self.assertGreaterEqual(summary["operation_count"], 1)
        self.assertGreaterEqual(summary["guard_count"], 1)
        self.assertTrue(summary["guard_samples"])
        self.assertEqual(summary["break_reasons"], [])

    def test_intentional_break_is_reported_and_fullgraph_rejects_it(self) -> None:
        summary = compile_pipeline.explain_summary(
            compile_pipeline.intentional_graph_break,
            (torch.randn(3, 4),),
        )
        self.assertGreaterEqual(summary["graph_count"], 2)
        self.assertGreaterEqual(summary["graph_break_count"], 1)
        self.assertTrue(summary["break_reasons"])

        failure = compile_pipeline.fullgraph_failure_probe()
        self.assertEqual(failure["status"], "rejected_as_expected")
        self.assertTrue(failure["error_type"])
        self.assertIn("graph_break", failure["message"])

    def test_dynamic_true_accepts_two_batch_shapes(self) -> None:
        result = compile_pipeline.dynamic_shape_probe()
        self.assertEqual(result["backend"], "eager")
        self.assertEqual(result["input_batch_sizes"], [2, 5])
        self.assertEqual(result["output_shapes"], [[2], [5]])
        self.assertIn("dynamic=True", result["configuration"])
        self.assertIn("not asserted", result["claim"])

    def test_export_captures_ahead_of_time_graph_or_degrades_clearly(self) -> None:
        result = compile_pipeline.export_summary()
        if not result["available"]:
            self.assertIn("unavailable", result["reason"])
            self.skipTest(result["reason"])
        self.assertGreater(result["node_count"], 0)
        self.assertIn("graph_signature", result)
        self.assertGreaterEqual(result["range_constraint_count"], 0)

    def test_invalid_inputs_and_backend_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            compile_pipeline.make_inputs(0)
        with self.assertRaisesRegex(RuntimeError, "backend='not-a-real-backend'"):
            compile_pipeline.compare_eager_and_compile("not-a-real-backend")

    def test_integrated_report_makes_no_performance_claim(self) -> None:
        result = compile_pipeline.run_checks()
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(
            result["scope"],
            "correctness_and_diagnostics_only_no_performance_claim",
        )
        self.assertEqual(
            result["fullgraph_failure"]["status"],
            "rejected_as_expected",
        )


if __name__ == "__main__":
    unittest.main()
