"""CPU-first tests for benchmark_profiler.py; no speed threshold is asserted."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import benchmark_profiler  # noqa: E402


class BenchmarkProfilerCpuTest(unittest.TestCase):
    def test_cpu_report_matches_schema_and_keeps_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = benchmark_profiler.build_report(
                "cpu",
                temporary,
                benchmark_config=benchmark_profiler.BenchmarkConfig(
                    rows=32,
                    columns=32,
                    min_run_time=0.01,
                ),
                profiler_config=benchmark_profiler.ProfilerConfig(
                    wait=1,
                    warmup=1,
                    active=1,
                    repeat=1,
                ),
            )

            benchmark_profiler.validate_report(report)
            self.assertEqual(
                set(report), set(benchmark_profiler.REPORT_SCHEMA["required"])
            )
            self.assertEqual(
                report["scope"], "local_observation_not_general_conclusion"
            )
            self.assertTrue(report["protocol"]["correctness_before_performance"])
            self.assertIn("not run", report["protocol"]["degradation"])

            suite = report["benchmarks"][0]
            self.assertTrue(suite["correctness"]["passed"])
            self.assertEqual(suite["controlled_variable"], "implementation only")
            self.assertEqual(len(suite["measurements"]), 2)
            self.assertEqual(suite["correctness"]["layout"], "torch.strided")
            self.assertTrue(suite["correctness"]["contiguous"])
            self.assertEqual(suite["correctness"]["value_range"], [-3.0, 3.0])
            self.assertIn("blocked_autorange", report["protocol"]["warmup_policy"])
            self.assertIn("not controlled", report["protocol"]["background_load"])
            for measurement in suite["measurements"]:
                self.assertGreaterEqual(measurement["replicates"], 3)
                self.assertEqual(
                    measurement["replicates"],
                    len(measurement["seconds_per_call"]),
                )
                self.assertGreaterEqual(measurement["median_seconds"], 0.0)
                self.assertGreaterEqual(measurement["iqr_seconds"], 0.0)

    def test_scheduled_profiler_exports_trace_with_record_function_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observation = benchmark_profiler.run_profiler(
                "cpu",
                temporary,
                benchmark_profiler.ProfilerConfig(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1,
                    record_shapes=True,
                    with_stack=False,
                    profile_memory=True,
                ),
            )
            trace_path = Path(observation["trace"]["path"])
            self.assertTrue(trace_path.is_file())
            self.assertEqual(observation["activities"], ["CPU"])
            self.assertFalse(observation["cuda_activity_collected"])
            self.assertEqual(
                observation["schedule"],
                {"wait": 1, "warmup": 1, "active": 2, "repeat": 1, "total_steps": 4},
            )
            self.assertTrue(observation["trace"]["record_function_marker_present"])
            self.assertGreater(observation["trace"]["event_count"], 0)
            self.assertGreater(observation["trace"]["bytes"], 0)

            with trace_path.open("r", encoding="utf-8") as handle:
                trace = json.load(handle)
            names = {
                event.get("name")
                for event in trace["traceEvents"]
                if isinstance(event, dict)
            }
            self.assertIn("benchmark_profiler.step", names)
            self.assertIn("benchmark_profiler.forward", names)

    def test_cuda_fallback_is_explicit_without_executing_cuda(self) -> None:
        devices, message = benchmark_profiler.resolve_devices(
            "auto", cuda_available=False
        )
        self.assertEqual(devices, ("cpu",))
        self.assertIn("CUDA unavailable", message)
        self.assertIn("GPU timing", message)
        self.assertIn("CUDA activity", message)

    def test_invalid_config_and_report_paths_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_run_time must be positive"):
            benchmark_profiler.run_benchmark_suite(
                "cpu",
                benchmark_profiler.BenchmarkConfig(min_run_time=0),
            )
        with self.assertRaisesRegex(ValueError, "measurement_repeats"):
            benchmark_profiler.BenchmarkConfig(measurement_repeats=2).validate()
        with self.assertRaisesRegex(ValueError, "active must be positive"):
            benchmark_profiler.ProfilerConfig(active=0).validate()
        with self.assertRaisesRegex(ValueError, "repeat=1"):
            benchmark_profiler.ProfilerConfig(repeat=2).validate()
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            benchmark_profiler.resolve_devices("cuda", cuda_available=False)

        with tempfile.TemporaryDirectory() as temporary:
            report = benchmark_profiler.build_report(
                "cpu",
                temporary,
                benchmark_config=benchmark_profiler.BenchmarkConfig(
                    rows=16, columns=16, min_run_time=0.005
                ),
                profiler_config=benchmark_profiler.ProfilerConfig(
                    wait=0, warmup=1, active=1, repeat=1
                ),
            )
        missing_scope = copy.deepcopy(report)
        missing_scope.pop("scope")
        with self.assertRaisesRegex(
            benchmark_profiler.ReportValidationError, "keys mismatch"
        ):
            benchmark_profiler.validate_report(missing_scope)

        bad_distribution = copy.deepcopy(report)
        bad_distribution["benchmarks"][0]["measurements"][0][
            "seconds_per_call"
        ] = [0.001]
        with self.assertRaisesRegex(
            benchmark_profiler.ReportValidationError, "at least three samples"
        ):
            benchmark_profiler.validate_report(bad_distribution)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: optional activity test skipped")
class BenchmarkProfilerCudaTest(unittest.TestCase):
    def test_cuda_benchmark_synchronizes_and_profiler_collects_cuda_activity(self) -> None:
        suite = benchmark_profiler.run_benchmark_suite(
            "cuda",
            benchmark_profiler.BenchmarkConfig(
                rows=16, columns=16, min_run_time=0.005
            ),
        )
        self.assertTrue(suite["correctness"]["passed"])
        self.assertEqual(suite["correctness"]["device"], "cuda")

        with tempfile.TemporaryDirectory() as temporary:
            observation = benchmark_profiler.run_profiler(
                "cuda",
                temporary,
                benchmark_profiler.ProfilerConfig(
                    wait=1, warmup=1, active=1, repeat=1
                ),
            )
        self.assertEqual(observation["activities"], ["CPU", "CUDA"])
        self.assertTrue(observation["cuda_activity_collected"])


if __name__ == "__main__":
    unittest.main()
