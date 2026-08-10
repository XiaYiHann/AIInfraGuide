"""Correctness-first benchmark and profiler evidence for PyTorch lesson 4.8.

Run from the repository root:
    python3 examples/pytorch/benchmark_profiler.py --device cpu
    python3 examples/pytorch/benchmark_profiler.py --device auto

Every timing is a local observation of the current process and machine. The
script deliberately sets no speed threshold and makes no universal claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import benchmark


SCHEMA_VERSION = "1.0"
REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "scope",
        "environment",
        "protocol",
        "benchmarks",
        "profiler",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "scope": {"const": "local_observation_not_general_conclusion"},
        "environment": {"type": "object"},
        "protocol": {"type": "object"},
        "benchmarks": {"type": "array"},
        "profiler": {"type": "object"},
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed workload and measurement controls for one local comparison."""

    rows: int = 256
    columns: int = 256
    seed: int = 20260810
    num_threads: int = 1
    min_run_time: float = 0.05
    measurement_repeats: int = 5

    def validate(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.min_run_time <= 0:
            raise ValueError("min_run_time must be positive")
        if self.measurement_repeats < 3:
            raise ValueError("measurement_repeats must be at least 3")


@dataclass(frozen=True)
class ProfilerConfig:
    """Schedule and optional high-overhead evidence toggles."""

    wait: int = 1
    warmup: int = 1
    active: int = 2
    repeat: int = 1
    record_shapes: bool = False
    with_stack: bool = False
    profile_memory: bool = False

    def validate(self) -> None:
        if self.wait < 0 or self.warmup < 0:
            raise ValueError("profiler wait and warmup must be non-negative")
        if self.active <= 0:
            raise ValueError("profiler active must be positive")
        if self.repeat != 1:
            raise ValueError(
                "this single-trace course exporter requires profiler repeat=1; "
                "multi-cycle collection needs unique output paths"
            )

    @property
    def total_steps(self) -> int:
        return (self.wait + self.warmup + self.active) * self.repeat


class ReportValidationError(ValueError):
    """Raised when an experiment report violates the public report contract."""


def environment_report() -> dict[str, Any]:
    """Record enough local context to prevent timings from floating free."""
    cuda_available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_runtime_built": torch.version.cuda,
        "cuda_device": None,
    }
    if cuda_available:
        report["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return report


def resolve_devices(
    requested: str, *, cuda_available: bool | None = None
) -> tuple[tuple[str, ...], str]:
    """Resolve CPU/CUDA execution without pretending a skipped GPU path ran."""
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("requested device must be one of: cpu, cuda, auto")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cpu":
        return ("cpu",), "CPU requested; CUDA benchmark and CUDA profiler activity were not run."
    if requested == "cuda":
        return ("cuda",), "CUDA requested; CPU activity is still collected by the profiler."
    if available:
        return ("cpu", "cuda"), "CUDA available; CPU and CUDA local observations will run."
    return (
        ("cpu",),
        "CUDA unavailable: ran CPU benchmark and CPU profiler activity only; "
        "GPU timing and CUDA activity were not measured.",
    )


def _fixed_input(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Create the same deterministic values on every requested device."""
    config.validate()
    torch.manual_seed(config.seed)
    values = torch.linspace(
        -3.0,
        3.0,
        steps=config.rows * config.columns,
        dtype=torch.float32,
        device=device,
    )
    return values.reshape(config.rows, config.columns)


def check_implementations(input_tensor: torch.Tensor) -> dict[str, Any]:
    """Prove equivalence before allowing either implementation to be timed."""
    reference = torch.relu(input_tensor)
    candidate = input_tensor.clamp_min(0)
    if not torch.equal(reference, candidate):
        max_abs_error = (reference - candidate).abs().max().item()
        raise AssertionError(
            f"benchmark candidates are not equivalent; max_abs_error={max_abs_error}"
        )
    return {
        "passed": True,
        "criterion": "torch.equal(torch.relu(x), x.clamp_min(0))",
        "shape": list(input_tensor.shape),
        "dtype": str(input_tensor.dtype),
        "device": input_tensor.device.type,
        "layout": str(input_tensor.layout),
        "strides": list(input_tensor.stride()),
        "contiguous": input_tensor.is_contiguous(),
        "value_generator": "torch.linspace(-3.0, 3.0, inclusive)",
        "value_range": [-3.0, 3.0],
    }


def _measurement_record(
    measurements: list[benchmark.Measurement], *, device: str, statement: str
) -> dict[str, Any]:
    if not measurements:
        raise AssertionError("Timer returned no measurements")
    per_call = [
        float(value)
        for measurement in measurements
        for value in measurement.times
    ]
    if len(per_call) < 3 or any(value < 0 for value in per_call):
        raise AssertionError("Timer must return at least three non-negative samples")
    first = measurements[0]
    q1, _, q3 = statistics.quantiles(per_call, n=4, method="inclusive")
    return {
        "device": device,
        "statement": statement,
        "label": first.task_spec.label,
        "sub_label": first.task_spec.sub_label,
        "description": first.task_spec.description,
        "number_per_run": [measurement.number_per_run for measurement in measurements],
        "replicates": len(per_call),
        "seconds_per_call": per_call,
        "median_seconds": statistics.median(per_call),
        "mean_seconds": statistics.fmean(per_call),
        "iqr_seconds": q3 - q1,
        "min_seconds": min(per_call),
        "max_seconds": max(per_call),
    }


def run_benchmark_suite(
    device_name: str,
    config: BenchmarkConfig | None = None,
    *,
    print_compare: bool = False,
) -> dict[str, Any]:
    """Run blocked_autorange and retain the complete local distribution."""
    config = config or BenchmarkConfig()
    config.validate()
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("benchmark device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CUDA benchmark")

    device = torch.device(device_name)
    input_tensor = _fixed_input(config, device)
    correctness = check_implementations(input_tensor)
    if device.type == "cuda":
        # Clear input construction from the benchmark boundary. Timer itself
        # synchronizes asynchronous accelerator work when it measures a block.
        torch.cuda.synchronize(device)

    specs = (
        ("torch.relu(input_tensor)", "torch.relu"),
        ("input_tensor.clamp_min(0)", "Tensor.clamp_min"),
    )
    compare_measurements: list[benchmark.Measurement] = []
    records: list[dict[str, Any]] = []
    for statement, description in specs:
        timer = benchmark.Timer(
            stmt=statement,
            globals={"torch": torch, "input_tensor": input_tensor},
            label="fixed-shape elementwise activation",
            sub_label=f"{config.rows}x{config.columns} float32 on {device.type}",
            description=description,
            env=f"{platform.system()} | torch {torch.__version__}",
            num_threads=config.num_threads,
        )
        repeated = [
            timer.blocked_autorange(min_run_time=config.min_run_time)
            for _ in range(config.measurement_repeats)
        ]
        compare_measurements.append(repeated[0])
        records.append(
            _measurement_record(
                repeated, device=device.type, statement=statement
            )
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if print_compare:
        benchmark.Compare(compare_measurements).print()

    return {
        "correctness": correctness,
        "config": asdict(config),
        "controlled_variable": "implementation only",
        "measurements": records,
        "interpretation": (
            "Local observations only. Compare medians together with IQR and the "
            "full per-call distribution; do not generalize to another workload."
        ),
    }


def _trace_summary(trace_path: Path) -> dict[str, Any]:
    """Validate the exported Chrome trace and report structural evidence."""
    if not trace_path.is_file():
        raise AssertionError(f"profiler did not create trace: {trace_path}")
    with trace_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise AssertionError("Chrome trace root must be a JSON object")
    events = payload.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise AssertionError("Chrome trace must contain a non-empty traceEvents list")
    names = {
        event.get("name")
        for event in events
        if isinstance(event, Mapping) and isinstance(event.get("name"), str)
    }
    marker_present = "benchmark_profiler.step" in names
    if not marker_present:
        raise AssertionError("record_function marker is missing from the trace")
    return {
        "path": str(trace_path),
        "bytes": trace_path.stat().st_size,
        "event_count": len(events),
        "record_function_marker_present": marker_present,
    }


def run_profiler(
    device_name: str,
    output_dir: str | Path,
    config: ProfilerConfig | None = None,
) -> dict[str, Any]:
    """Export one scheduled trace with explicit CPU/CUDA activity degradation."""
    config = config or ProfilerConfig()
    config.validate()
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("profiler device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA profiler activity")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / f"benchmark-profiler-{device_name}.json"
    if trace_path.suffix != ".json":
        raise ValueError("profiler trace path must end in .json")
    if trace_path.exists():
        trace_path.unlink()

    device = torch.device(device_name)
    torch.manual_seed(20260810)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 16)).to(
        device
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    inputs = torch.linspace(-1.0, 1.0, steps=8 * 64, device=device).reshape(8, 64)

    # A finite eager result is the correctness gate before profiler overhead is
    # introduced. Profiling is diagnostic instrumentation, not validation.
    with torch.no_grad():
        eager_output = model(inputs)
    if not bool(torch.isfinite(eager_output).all().item()):
        raise AssertionError("profiler workload failed the eager correctness gate")

    activities = [ProfilerActivity.CPU]
    activity_names = ["CPU"]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        activity_names.append("CUDA")
        torch.cuda.synchronize(device)

    exported: list[str] = []

    def export_ready(profiler: torch.profiler.profile) -> None:
        profiler.export_chrome_trace(str(trace_path))
        exported.append(str(trace_path))

    schedule = torch.profiler.schedule(
        wait=config.wait,
        warmup=config.warmup,
        active=config.active,
        repeat=config.repeat,
    )
    with profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=export_ready,
        record_shapes=config.record_shapes,
        with_stack=config.with_stack,
        profile_memory=config.profile_memory,
        acc_events=True,
    ) as profiler:
        for _ in range(config.total_steps):
            with record_function("benchmark_profiler.step"):
                optimizer.zero_grad(set_to_none=True)
                with record_function("benchmark_profiler.forward"):
                    output = model(inputs)
                    loss = output.square().mean()
                with record_function("benchmark_profiler.backward"):
                    loss.backward()
                with record_function("benchmark_profiler.optimizer"):
                    optimizer.step()
            profiler.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if len(exported) != config.repeat:
        raise AssertionError(
            f"expected {config.repeat} exported trace(s), got {len(exported)}"
        )
    trace = _trace_summary(trace_path)
    operator_names = sorted(
        {
            event.key
            for event in profiler.key_averages()
            if isinstance(event.key, str)
            and (
                event.key.startswith("aten::")
                or event.key.startswith("autograd::")
                or event.key.startswith("benchmark_profiler.")
            )
        }
    )

    return {
        "device": device.type,
        "activities": activity_names,
        "cuda_activity_collected": "CUDA" in activity_names,
        "schedule": {
            "wait": config.wait,
            "warmup": config.warmup,
            "active": config.active,
            "repeat": config.repeat,
            "total_steps": config.total_steps,
        },
        "evidence_options": {
            "record_shapes": config.record_shapes,
            "with_stack": config.with_stack,
            "profile_memory": config.profile_memory,
            "overhead_warning": (
                "Shape, stack, and memory evidence increase collection overhead and "
                "trace volume; compare an instrumented run with an uninstrumented "
                "benchmark instead of treating profiler time as baseline time."
            ),
        },
        "trace": trace,
        "recorded_operator_names": operator_names,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate the stable top-level schema and key measurement invariants."""
    if not isinstance(report, Mapping):
        raise ReportValidationError("report must be a mapping")
    required = set(REPORT_SCHEMA["required"])
    missing = required.difference(report)
    extra = set(report).difference(required)
    if missing or extra:
        raise ReportValidationError(
            f"report keys mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    if report["schema_version"] != SCHEMA_VERSION:
        raise ReportValidationError("unsupported schema_version")
    if report["scope"] != "local_observation_not_general_conclusion":
        raise ReportValidationError("report scope must reject universal conclusions")
    if not isinstance(report["benchmarks"], list) or not report["benchmarks"]:
        raise ReportValidationError("benchmarks must be a non-empty list")
    for suite in report["benchmarks"]:
        if not suite.get("correctness", {}).get("passed"):
            raise ReportValidationError("every benchmark needs a correctness gate")
        measurements = suite.get("measurements")
        if not isinstance(measurements, list) or len(measurements) < 2:
            raise ReportValidationError("each comparison needs at least two measurements")
        for measurement in measurements:
            distribution = measurement.get("seconds_per_call")
            if not isinstance(distribution, list) or len(distribution) < 3:
                raise ReportValidationError(
                    "measurement distribution needs at least three samples"
                )
            if any(not isinstance(value, (int, float)) or value < 0 for value in distribution):
                raise ReportValidationError("measurement distribution is invalid")
    trace = report["profiler"].get("trace", {})
    if not trace.get("record_function_marker_present"):
        raise ReportValidationError("profiler trace lacks the record_function marker")


def build_report(
    requested_device: str,
    output_dir: str | Path,
    *,
    benchmark_config: BenchmarkConfig | None = None,
    profiler_config: ProfilerConfig | None = None,
    print_compare: bool = False,
) -> dict[str, Any]:
    """Run correctness, benchmark, and profiler stages in that order."""
    devices, degradation = resolve_devices(requested_device)
    suites = [
        run_benchmark_suite(
            device,
            benchmark_config,
            print_compare=print_compare,
        )
        for device in devices
    ]
    profiler_device = "cuda" if "cuda" in devices else "cpu"
    profiler_report = run_profiler(profiler_device, output_dir, profiler_config)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "local_observation_not_general_conclusion",
        "environment": environment_report(),
        "protocol": {
            "correctness_before_performance": True,
            "fixed_input": True,
            "single_controlled_variable": "implementation only",
            "timer": "torch.utils.benchmark.Timer.blocked_autorange",
            "statistics": "median, mean, IQR, min, max, and full per-call samples",
            "warmup_policy": "delegated to every blocked_autorange repetition",
            "background_load": "not controlled; record the environment and rerun if noisy",
            "cuda_completion": (
                "Timer accelerator synchronization plus explicit suite/profiler boundaries"
            ),
            "degradation": degradation,
        },
        "benchmarks": suites,
        "profiler": profiler_report,
    }
    validate_report(report)
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "aiinfraguide-pytorch-4.8",
        help="Directory for the Chrome trace and report (default: system temp dir).",
    )
    parser.add_argument("--min-run-time", type=float, default=0.05)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    args = parser.parse_args()

    benchmark_config = BenchmarkConfig(min_run_time=args.min_run_time)
    profiler_config = ProfilerConfig(
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        profile_memory=args.profile_memory,
    )
    report = build_report(
        args.device,
        args.output_dir,
        benchmark_config=benchmark_config,
        profiler_config=profiler_config,
        print_compare=True,
    )
    report_path = args.output_dir / "benchmark-profiler-report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    print(f"report: {report_path}")
    print(f"trace: {report['profiler']['trace']['path']}")
    print("all benchmark-profiler checks passed; timings are local observations only")


if __name__ == "__main__":
    main()
