"""Correctness-first CUDA stream, event, and timing experiments for lesson 4.6.

Run from the repository root:
    python3 examples/pytorch/cuda_async.py

The script degrades explicitly when CUDA is unavailable. CUDA measurements are
local observations of the timing method, not speedup or throughput claims.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimingObservation:
    """One correctly bounded local timing observation."""

    warmup_iterations: int
    measured_iterations: int
    elements: int
    enqueue_seconds: float
    completion_was_pending_after_enqueue: bool
    event_milliseconds: float
    synchronized_wall_seconds: float


@dataclass(frozen=True)
class StreamObservation:
    """Facts observed while switching from the current stream to a side stream."""

    current_was_default: bool
    context_selected_side_stream: bool
    context_restored_previous_stream: bool


@dataclass(frozen=True)
class H2DObservation:
    """Correctness evidence for one pinned, non-blocking H2D transfer."""

    source_was_pinned: bool
    copy_used_non_default_stream: bool
    copied_values: list[int]


def environment_report() -> dict[str, str | bool | int | None]:
    """Return the environment fields needed to interpret a CUDA smoke run."""
    available = torch.cuda.is_available()
    report: dict[str, str | bool | int | None] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime_built": torch.version.cuda,
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "device": None,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
    }
    if available:
        report["device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return report


def cpu_fallback_status(cuda_available: bool | None = None) -> str:
    """Describe the explicit CPU-only degradation without implying GPU testing."""
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if available:
        return "CUDA is available; CUDA correctness smoke can run."
    return (
        "CUDA unavailable: skipped streams, Events, pinned H2D, and GPU timing; "
        "GPU performance was not measured."
    )


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    return torch.device("cuda", torch.cuda.current_device())


def observe_stream_context() -> StreamObservation:
    """Prove that current means the stream selected for the present context."""
    device = _cuda_device()
    previous = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    side = torch.cuda.Stream(device=device)

    with torch.cuda.stream(side):
        selected = torch.cuda.current_stream(device) == side

    return StreamObservation(
        current_was_default=previous == default,
        context_selected_side_stream=selected,
        context_restored_previous_stream=torch.cuda.current_stream(device) == previous,
    )


def observe_enqueue_and_timing(
    *,
    elements: int = 1 << 20,
    warmup_iterations: int = 3,
    measured_iterations: int = 8,
) -> TimingObservation:
    """Contrast host enqueue with completion and time work with CUDA Events.

    No ratio or minimum runtime is asserted: launch latency and kernel duration
    depend on the local software stack and GPU. Warmup is outside the measured
    interval, and both Event and wall-clock intervals have explicit completion
    boundaries.
    """
    if elements <= 0:
        raise ValueError("elements must be positive")
    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError(
            "warmup_iterations must be >= 0 and measured_iterations must be > 0"
        )

    device = _cuda_device()
    value = torch.linspace(-1.0, 1.0, elements, device=device)

    # Warm up lazy CUDA/library initialization before recording the sample.
    for _ in range(warmup_iterations):
        value = torch.sin(value).add_(0.001)
    torch.cuda.synchronize(device)

    # This interval ends after commands are enqueued. The Event query is only
    # an observation: a fast GPU is allowed to have completed already.
    enqueue_done = torch.cuda.Event()
    enqueue_start = time.perf_counter()
    queued_value = torch.sin(value).add_(0.001)
    enqueue_done.record()
    enqueue_seconds = time.perf_counter() - enqueue_start
    completion_was_pending = not enqueue_done.query()
    enqueue_done.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(measured_iterations):
        queued_value = torch.sin(queued_value).add_(0.001)
    end.record()
    end.synchronize()  # Event timestamps are valid only after completion.
    synchronized_wall_seconds = time.perf_counter() - wall_start
    event_milliseconds = start.elapsed_time(end)

    # Materialize one value only after timing. This proves queued work produced
    # a finite result without inserting a hidden host sync inside the interval.
    if not torch.isfinite(queued_value[0].cpu()):
        raise AssertionError("timed CUDA work produced a non-finite value")

    return TimingObservation(
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        elements=elements,
        enqueue_seconds=enqueue_seconds,
        completion_was_pending_after_enqueue=completion_was_pending,
        event_milliseconds=event_milliseconds,
        synchronized_wall_seconds=synchronized_wall_seconds,
    )


def event_dependency_pipeline(length: int = 64) -> torch.Tensor:
    """Run producer and consumer streams with explicit Event/lifetime edges."""
    if length <= 0:
        raise ValueError("length must be positive")

    device = _cuda_device()
    origin = torch.cuda.current_stream(device)
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    source = torch.arange(length, dtype=torch.int64, device=device)

    # source was created on origin. The producer must not read it before origin
    # reaches this point.
    producer.wait_stream(origin)
    ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        produced = source * 2
        ready.record()
    # source's Python lifetime may end before producer work does. Tell the
    # caching allocator that producer also uses its storage.
    source.record_stream(producer)

    with torch.cuda.stream(consumer):
        ready.wait()  # Future consumer work waits for the producer checkpoint.
        consumed = produced + 1
    produced.record_stream(consumer)

    # Rejoin the current stream only where the host result is actually needed.
    origin.wait_stream(consumer)
    consumed.record_stream(origin)
    result = consumed.cpu()  # Deliberate device-to-host synchronization point.
    expected = torch.arange(length, dtype=torch.int64).mul_(2).add_(1)
    if not torch.equal(result, expected):
        raise AssertionError("cross-stream dependency produced an incorrect result")
    return result


def non_blocking_h2d_pipeline(length: int = 64) -> H2DObservation:
    """Copy pinned host data on a side stream and consume it after an Event."""
    if length <= 0:
        raise ValueError("length must be positive")

    device = _cuda_device()
    current = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device=device)
    source = torch.arange(length, dtype=torch.int64, pin_memory=True)
    copied = None
    copied_event = torch.cuda.Event()

    with torch.cuda.stream(copy_stream):
        copied = source.to(device, non_blocking=True)
        copied_event.record()

    # Keep source alive and unmodified until the copy's Event has completed.
    current.wait_event(copied_event)
    copied.record_stream(current)
    result = copied.cpu()
    expected = torch.arange(length, dtype=torch.int64)
    if not torch.equal(result, expected):
        raise AssertionError("non-blocking H2D copy produced incorrect values")

    return H2DObservation(
        source_was_pinned=source.is_pinned(),
        copy_used_non_default_stream=copy_stream != torch.cuda.default_stream(device),
        copied_values=result.tolist(),
    )


def run_cuda_checks() -> dict[str, object]:
    """Run every CUDA-only correctness check used by the lesson."""
    if not torch.cuda.is_available():
        return {"status": cpu_fallback_status(False), "cuda_tested": False}

    stream = observe_stream_context()
    timing = observe_enqueue_and_timing()
    dependency = event_dependency_pipeline()
    h2d = non_blocking_h2d_pipeline()
    torch.cuda.synchronize()

    assert stream.context_selected_side_stream
    assert stream.context_restored_previous_stream
    assert dependency.tolist() == [2 * index + 1 for index in range(dependency.numel())]
    assert h2d.source_was_pinned
    assert h2d.copy_used_non_default_stream
    assert timing.event_milliseconds >= 0.0
    assert timing.synchronized_wall_seconds >= 0.0

    return {
        "status": "CUDA correctness smoke passed; no speedup claim was made.",
        "cuda_tested": True,
        "stream": stream,
        "timing": timing,
        "dependency_last_value": int(dependency[-1]),
        "h2d": h2d,
    }


def main() -> None:
    report = environment_report()
    print("environment:", report)
    result = run_cuda_checks()
    print("status:", result["status"])
    if not result["cuda_tested"]:
        print("all CUDA-async CPU fallback checks passed")
        return

    timing = result["timing"]
    assert isinstance(timing, TimingObservation)
    print("stream context:", result["stream"])
    print(
        "enqueue observation:",
        {
            "host_seconds": timing.enqueue_seconds,
            "completion_was_pending": timing.completion_was_pending_after_enqueue,
        },
    )
    print(
        "timing observation (local, not a benchmark):",
        {
            "warmup_iterations": timing.warmup_iterations,
            "measured_iterations": timing.measured_iterations,
            "elements": timing.elements,
            "event_milliseconds": timing.event_milliseconds,
            "synchronized_wall_seconds": timing.synchronized_wall_seconds,
        },
    )
    print("event dependency last value:", result["dependency_last_value"])
    print("pinned non-blocking H2D:", result["h2d"])
    print("all CUDA-async correctness checks passed")


if __name__ == "__main__":
    main()
