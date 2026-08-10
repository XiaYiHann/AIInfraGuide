"""Correctness-first memory accounting experiments for lesson 4.7.

Run from the repository root:
    python3 examples/pytorch/memory_accounting.py --device auto

The CPU path parses and aggregates an explicit tensor ledger. When CUDA is
available, the script also runs a small correctness smoke and checks allocator
invariants. It does not trigger OOM or report performance ratios.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn


MEMORY_CATEGORIES = (
    "parameters",
    "gradients",
    "optimizer_state",
    "activations",
    "temporary_buffers",
)


@dataclass(frozen=True)
class LedgerEntry:
    """One explicit tensor-like item in a logical training memory ledger."""

    category: str
    name: str
    numel: int
    bytes_per_element: int

    def __post_init__(self) -> None:
        if self.category not in MEMORY_CATEGORIES:
            raise ValueError(f"unknown memory category: {self.category!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("entry name must be a non-empty string")
        if isinstance(self.numel, bool) or not isinstance(self.numel, int):
            raise TypeError("numel must be an integer")
        if self.numel < 0:
            raise ValueError("numel must be non-negative")
        if (
            isinstance(self.bytes_per_element, bool)
            or not isinstance(self.bytes_per_element, int)
        ):
            raise TypeError("bytes_per_element must be an integer")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")

    @property
    def nbytes(self) -> int:
        return self.numel * self.bytes_per_element

    def to_record(self) -> dict[str, str | int]:
        return {
            "category": self.category,
            "name": self.name,
            "numel": self.numel,
            "bytes_per_element": self.bytes_per_element,
        }


@dataclass(frozen=True)
class CudaMemoryPoint:
    """Current allocator counters at one correctness boundary."""

    label: str
    allocated: int
    reserved: int


@dataclass(frozen=True)
class CudaMemoryObservation:
    """Small CUDA smoke evidence; values are bytes, not benchmark results."""

    points: tuple[CudaMemoryPoint, ...]
    allocated_at_peak_reset: int
    peak_immediately_after_reset: int
    peak_allocated: int
    peak_reserved: int
    loss_was_finite: bool
    parameter_changed: bool


def parse_ledger_records(records: Iterable[Mapping[str, Any]]) -> list[LedgerEntry]:
    """Validate serialized records and turn them into typed ledger entries."""
    entries: list[LedgerEntry] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"ledger record {index} must be a mapping")
        required = {"category", "name", "numel", "bytes_per_element"}
        missing = required.difference(record)
        extra = set(record).difference(required)
        if missing or extra:
            raise ValueError(
                f"ledger record {index} has missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        entries.append(
            LedgerEntry(
                category=record["category"],
                name=record["name"],
                numel=record["numel"],
                bytes_per_element=record["bytes_per_element"],
            )
        )
    return entries


def parse_ledger_json(payload: str) -> list[LedgerEntry]:
    """Parse a JSON list using the same strict ledger contract."""
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise TypeError("ledger JSON must contain a list of records")
    return parse_ledger_records(decoded)


def aggregate_ledger(entries: Iterable[LedgerEntry]) -> dict[str, int]:
    """Return byte totals for every category, including categories at zero."""
    totals = {category: 0 for category in MEMORY_CATEGORIES}
    for entry in entries:
        if not isinstance(entry, LedgerEntry):
            raise TypeError("aggregate_ledger expects LedgerEntry values")
        totals[entry.category] += entry.nbytes
    return totals


def total_ledger_bytes(entries: Iterable[LedgerEntry]) -> int:
    return sum(entry.nbytes for entry in entries)


def tensor_entry(category: str, name: str, tensor: torch.Tensor) -> LedgerEntry:
    """Describe the logical bytes exposed by one Tensor."""
    return LedgerEntry(
        category=category,
        name=name,
        numel=tensor.numel(),
        bytes_per_element=tensor.element_size(),
    )


def _tensor_values(value: Any, prefix: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Walk nested optimizer state without assuming a particular optimizer."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _tensor_values(nested, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _tensor_values(nested, f"{prefix}.{index}")


def build_training_ledger(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    activations: Mapping[str, torch.Tensor] | None = None,
    temporary_buffers: Mapping[str, torch.Tensor] | None = None,
) -> list[LedgerEntry]:
    """Inventory explicit live tensors; this is not an allocator snapshot.

    Parameters and gradients come from the model, optimizer tensors come from
    its materialized state, and activations/temporary buffers must be supplied
    explicitly by the caller. The function intentionally makes omissions
    visible instead of guessing hidden framework or library allocations.
    """
    entries: list[LedgerEntry] = []
    parameter_names: dict[int, str] = {}

    for name, parameter in model.named_parameters():
        parameter_names[id(parameter)] = name
        entries.append(tensor_entry("parameters", name, parameter))
        if parameter.grad is not None:
            entries.append(tensor_entry("gradients", f"{name}.grad", parameter.grad))

    for parameter, state in optimizer.state.items():
        parameter_name = parameter_names.get(id(parameter), f"parameter_{id(parameter)}")
        for state_name, tensor in _tensor_values(state, parameter_name):
            entries.append(tensor_entry("optimizer_state", state_name, tensor))

    for name, tensor in (activations or {}).items():
        entries.append(tensor_entry("activations", name, tensor))
    for name, tensor in (temporary_buffers or {}).items():
        entries.append(tensor_entry("temporary_buffers", name, tensor))

    return entries


def cpu_ledger_demo() -> list[LedgerEntry]:
    """Materialize all five categories on CPU and return a parseable ledger."""
    torch.manual_seed(20260810)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    targets = torch.tensor([0, 1, 0])

    hidden = model[0](inputs)
    activated = model[1](hidden)
    output = model[2](activated)
    loss = nn.functional.cross_entropy(output, targets)
    loss.backward()
    optimizer.step()  # Materializes optimizer state; gradients remain explicit.

    scratch = torch.empty(16, dtype=torch.float32)
    entries = build_training_ledger(
        model,
        optimizer,
        activations={"hidden": hidden, "output": output},
        temporary_buffers={"scratch": scratch},
    )

    # Round-trip through JSON to prove the ledger is serialized data rather
    # than a print-only object graph.
    payload = json.dumps([entry.to_record() for entry in entries])
    parsed = parse_ledger_json(payload)
    if parsed != entries:
        raise AssertionError("ledger JSON round-trip changed an entry")
    return parsed


def cuda_fallback_status(cuda_available: bool | None = None) -> str:
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if available:
        return "CUDA is available; the small memory correctness smoke can run."
    return (
        "CUDA unavailable: skipped peak reset and allocated/reserved CUDA checks; "
        "GPU memory behavior and performance were not measured."
    )


def _memory_point(label: str, device: torch.device) -> CudaMemoryPoint:
    return CudaMemoryPoint(
        label=label,
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
    )


def run_cuda_memory_smoke() -> CudaMemoryObservation:
    """Run a small forward/backward/update and check allocator invariants.

    The experiment allocates only a tiny model and never attempts to trigger an
    OOM. Cleanup calls empty_cache only after all live experiment tensors leave
    scope; it is not used as a way to free live tensors.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CUDA memory smoke")

    device = torch.device("cuda", torch.cuda.current_device())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    def run_workload() -> CudaMemoryObservation:
        torch.cuda.reset_peak_memory_stats(device)
        allocated_at_reset = torch.cuda.memory_allocated(device)
        peak_after_reset = torch.cuda.max_memory_allocated(device)
        points = [_memory_point("after_peak_reset", device)]

        torch.manual_seed(20260810)
        torch.cuda.manual_seed_all(20260810)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model[0].weight.detach().clone()
        points.append(_memory_point("after_model", device))

        inputs = torch.randn(8, 16, device=device)
        targets = torch.arange(8, device=device) % 4
        output = model(inputs)
        loss = nn.functional.cross_entropy(output, targets)
        points.append(_memory_point("after_forward", device))

        loss.backward()
        points.append(_memory_point("after_backward", device))
        optimizer.step()
        torch.cuda.synchronize(device)
        points.append(_memory_point("after_optimizer_step", device))

        observation = CudaMemoryObservation(
            points=tuple(points),
            allocated_at_peak_reset=allocated_at_reset,
            peak_immediately_after_reset=peak_after_reset,
            peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),
            loss_was_finite=bool(torch.isfinite(loss.detach()).item()),
            parameter_changed=not torch.equal(before, model[0].weight.detach()),
        )

        if observation.peak_immediately_after_reset != observation.allocated_at_peak_reset:
            raise AssertionError("peak reset did not reset max allocated to current allocated")
        if not observation.loss_was_finite or not observation.parameter_changed:
            raise AssertionError("CUDA training correctness smoke failed")
        for point in observation.points:
            if point.allocated > point.reserved:
                raise AssertionError(f"allocated exceeded reserved at {point.label}")
        if observation.peak_allocated < max(point.allocated for point in points):
            raise AssertionError("peak allocated is below an observed current allocation")
        if observation.peak_reserved < max(point.reserved for point in points):
            raise AssertionError("peak reserved is below an observed current reservation")
        return observation

    try:
        return run_workload()
    finally:
        # Workload locals are gone here. This only returns unused cached blocks
        # to CUDA; it cannot free tensors that remain live elsewhere.
        gc.collect()
        torch.cuda.empty_cache()


def format_bytes(value: int) -> str:
    if value < 0:
        raise ValueError("byte value must be non-negative")
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="CPU always runs; auto additionally runs CUDA when available.",
    )
    args = parser.parse_args()

    entries = cpu_ledger_demo()
    totals = aggregate_ledger(entries)
    print("CPU logical ledger:")
    for category in MEMORY_CATEGORIES:
        print(f"  {category}: {totals[category]} bytes ({format_bytes(totals[category])})")
    print(f"  total: {total_ledger_bytes(entries)} bytes")
    print("CPU ledger JSON round-trip passed")

    if args.device == "cpu":
        print("CUDA path not requested; GPU memory behavior and performance were not measured.")
        print("all memory-accounting CPU checks passed")
        return

    if not torch.cuda.is_available():
        print(cuda_fallback_status(False))
        print("all memory-accounting CPU fallback checks passed")
        return

    observation = run_cuda_memory_smoke()
    print("CUDA allocator observations (bytes; correctness smoke, not a benchmark):")
    for point in observation.points:
        print(
            f"  {point.label}: allocated={point.allocated}, reserved={point.reserved}"
        )
    print(
        "  peaks:",
        {
            "allocated": observation.peak_allocated,
            "reserved": observation.peak_reserved,
            "reset_peak_equals_current": (
                observation.peak_immediately_after_reset
                == observation.allocated_at_peak_reset
            ),
        },
    )
    print("all memory-accounting CUDA correctness checks passed")


if __name__ == "__main__":
    main()
