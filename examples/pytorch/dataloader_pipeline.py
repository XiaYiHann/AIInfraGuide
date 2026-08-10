"""Runnable Dataset/DataLoader experiments for lesson 4.4.

The default command is CPU-only and does not download data:
    python3 examples/pytorch/dataloader_pipeline.py

Use ``--device auto`` to exercise the CUDA transfer path when available. The
script reports capability and correctness only; it does not claim H2D speedups.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
    get_worker_info,
)


class MapSequenceDataset(Dataset[dict[str, Any]]):
    """Small random-access dataset with variable-length token sequences."""

    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        self.sequences = [list(sequence) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        tokens = torch.tensor(self.sequences[index], dtype=torch.long)
        return {"id": index, "tokens": tokens, "target": index % 2}


class ShardedRangeDataset(IterableDataset[int]):
    """Stream integers once, explicitly sharding each worker replica."""

    def __init__(self, start: int, end: int) -> None:
        if end < start:
            raise ValueError("end must be greater than or equal to start")
        self.start = start
        self.end = end

    def __iter__(self) -> Iterator[int]:
        worker = get_worker_info()
        if worker is None:
            yield from range(self.start, self.end)
            return

        # Each worker owns one strided shard. Without this branch every worker
        # would iterate over the full range and return duplicate records.
        yield from range(self.start + worker.id, self.end, worker.num_workers)


class OrderedSampler(Sampler[int]):
    """Yield a caller-provided order of map-style dataset indices."""

    def __init__(self, order: Sequence[int]) -> None:
        self.order = list(order)

    def __iter__(self) -> Iterator[int]:
        yield from self.order

    def __len__(self) -> int:
        return len(self.order)


@dataclass
class PaddedBatch:
    """Custom collate result that supports DataLoader memory pinning."""

    ids: torch.Tensor
    tokens: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor

    def pin_memory(self) -> "PaddedBatch":
        """Pin every Tensor field; DataLoader calls this for custom batches."""
        return PaddedBatch(
            ids=self.ids.pin_memory(),
            tokens=self.tokens.pin_memory(),
            lengths=self.lengths.pin_memory(),
            targets=self.targets.pin_memory(),
        )

    def to(self, device: torch.device, *, non_blocking: bool) -> "PaddedBatch":
        """Move every Tensor field with one explicit transfer policy."""
        return PaddedBatch(
            ids=self.ids.to(device, non_blocking=non_blocking),
            tokens=self.tokens.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
        )


def pad_collate(samples: list[dict[str, Any]]) -> PaddedBatch:
    """Pad variable-length token sequences and retain their true lengths."""
    if not samples:
        raise ValueError("pad_collate requires at least one sample")

    token_tensors = [sample["tokens"] for sample in samples]
    if any(tokens.ndim != 1 or tokens.numel() == 0 for tokens in token_tensors):
        raise ValueError("every sample must contain a non-empty 1-D token tensor")

    lengths = torch.tensor([tokens.numel() for tokens in token_tensors])
    padded = torch.zeros((len(samples), int(lengths.max().item())), dtype=torch.long)
    for row, tokens in enumerate(token_tensors):
        padded[row, : tokens.numel()] = tokens

    return PaddedBatch(
        ids=torch.tensor([sample["id"] for sample in samples], dtype=torch.long),
        tokens=padded,
        lengths=lengths,
        targets=torch.tensor(
            [sample["target"] for sample in samples], dtype=torch.long
        ),
    )


class SeedProbeDataset(Dataset[dict[str, int]]):
    """Expose worker identity and RNG values for reproducibility tests."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, int]:
        worker = get_worker_info()
        return {
            "index": index,
            "worker_id": -1 if worker is None else worker.id,
            "torch_seed": torch.initial_seed(),
            "python_random": random.randrange(0, 2**31),
        }


def first_sample(samples: list[dict[str, int]]) -> dict[str, int]:
    """Keep Python integers intact for the one-sample RNG probe batches."""
    if len(samples) != 1:
        raise ValueError("first_sample requires batch_size=1")
    return samples[0]


def seed_worker(worker_id: int) -> None:
    """Seed Python RNG from the unique PyTorch seed assigned to this worker."""
    worker = get_worker_info()
    if worker is None or worker.id != worker_id:
        raise RuntimeError("seed_worker must run inside its matching worker")
    random.seed(worker.seed)


def collect_seed_records(num_workers: int = 2) -> list[tuple[int, int, int, int]]:
    """Collect a deterministic trace of worker IDs and random values."""
    if num_workers == 0:
        # There is no worker_init_fn on the single-process path, so seed the
        # Python RNG that __getitem__ uses in the main process explicitly.
        random.seed(20260810)
    generator = torch.Generator().manual_seed(20260810)
    loader = DataLoader(
        SeedProbeDataset(),
        batch_size=1,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        collate_fn=first_sample,
    )
    records: list[tuple[int, int, int, int]] = []
    for sample in loader:
        records.append(
            (
                int(sample["index"]),
                int(sample["worker_id"]),
                int(sample["torch_seed"]),
                int(sample["python_random"]),
            )
        )
    return records


def sampler_batches(*, drop_last: bool) -> list[list[int]]:
    """Show how Sampler order becomes BatchSampler groups."""
    dataset = MapSequenceDataset([[10], [20, 21], [30], [40, 41, 42], [50]])
    sampler = OrderedSampler([4, 2, 0, 3, 1])
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=drop_last)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=pad_collate)
    return [batch.ids.tolist() for batch in loader]


def distributed_sampler_entry() -> tuple[list[int], list[int]]:
    """Demonstrate rank-local index partitioning without starting DDP."""
    dataset = MapSequenceDataset([[index] for index in range(8)])
    rank0 = DistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=False, drop_last=False
    )
    rank1 = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=False, drop_last=False
    )
    rank0.set_epoch(0)
    rank1.set_epoch(0)
    return list(rank0), list(rank1)


def invalid_sampler_configuration() -> str:
    """Return the expected error for mutually exclusive shuffle and sampler."""
    dataset = MapSequenceDataset([[1], [2]])
    try:
        DataLoader(
            dataset,
            shuffle=True,
            sampler=OrderedSampler([0, 1]),
        )
    except ValueError as error:
        return str(error)
    raise AssertionError("DataLoader should reject shuffle together with sampler")


def transfer_batch(
    batch: PaddedBatch,
    *,
    request_cuda: bool,
    cuda_available: bool | None = None,
) -> tuple[PaddedBatch, str]:
    """Move one batch or explicitly degrade to CPU without speed claims."""
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if not request_cuda:
        return batch, "CPU path selected; H2D and pin-memory performance not measured."
    if not available:
        return (
            batch,
            "CUDA unavailable: kept the batch on CPU; pin-memory and H2D "
            "performance were not measured.",
        )

    moved = batch.to(torch.device("cuda"), non_blocking=True)
    torch.cuda.synchronize()
    return (
        moved,
        "CUDA transfer completed for correctness; no pin-memory or H2D "
        "performance claim was made.",
    )


def measure_loader(
    loader: DataLoader[PaddedBatch],
    *,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, float | int]:
    """Measure a fixed number of loader waits; this is not a GPU benchmark."""
    if warmup_batches < 0 or measured_batches <= 0:
        raise ValueError("warmup_batches must be >= 0 and measured_batches must be > 0")

    iterator = iter(loader)
    for _ in range(warmup_batches):
        next(iterator)

    samples = 0
    start = time.perf_counter()
    for _ in range(measured_batches):
        batch = next(iterator)
        samples += int(batch.ids.numel())
    elapsed = time.perf_counter() - start
    return {
        "batches": measured_batches,
        "samples": samples,
        "seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


def make_demo_loader(
    num_workers: int,
    *,
    pin_memory: bool,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader[PaddedBatch]:
    """Build a finite loader with valid worker/prefetch option combinations."""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    sequences = [list(range(1, 2 + index % 5)) for index in range(64)]
    options: dict[str, Any] = {
        "dataset": MapSequenceDataset(sequences),
        "batch_size": 4,
        "num_workers": num_workers,
        "collate_fn": pad_collate,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": torch.Generator().manual_seed(20260810),
        "drop_last": True,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["persistent_workers"] = persistent_workers
    return DataLoader(**options)


def compare_loader_configs(*, pin_memory: bool) -> list[dict[str, Any]]:
    """Observe fixed work under three policies without imposing speed thresholds."""
    configs = [
        {"num_workers": 0, "prefetch_factor": None, "persistent_workers": False},
        {"num_workers": 2, "prefetch_factor": 1, "persistent_workers": False},
        {"num_workers": 2, "prefetch_factor": 2, "persistent_workers": True},
    ]
    observations: list[dict[str, Any]] = []
    for config in configs:
        loader = make_demo_loader(
            config["num_workers"],
            pin_memory=pin_memory,
            prefetch_factor=config["prefetch_factor"] or 2,
            persistent_workers=config["persistent_workers"],
        )
        metrics = measure_loader(loader, warmup_batches=1, measured_batches=4)
        observations.append({**config, "pin_memory": pin_memory, **metrics})
        del loader
    return observations


def run_checks(num_workers: int = 2) -> None:
    """Assert the CPU invariants demonstrated by the lesson."""
    dataset = MapSequenceDataset([[1, 2], [3], [4, 5, 6]])
    batch = pad_collate([dataset[0], dataset[1]])
    assert batch.ids.tolist() == [0, 1]
    assert batch.tokens.tolist() == [[1, 2], [3, 0]]
    assert batch.lengths.tolist() == [2, 1]

    assert sampler_batches(drop_last=False) == [[4, 2], [0, 3], [1]]
    assert sampler_batches(drop_last=True) == [[4, 2], [0, 3]]

    stream_values = list(
        DataLoader(
            ShardedRangeDataset(0, 12),
            batch_size=None,
            num_workers=num_workers,
        )
    )
    assert sorted(stream_values) == list(range(12))
    assert len(stream_values) == len(set(stream_values))

    first_seed_trace = collect_seed_records(num_workers=num_workers)
    second_seed_trace = collect_seed_records(num_workers=num_workers)
    assert first_seed_trace == second_seed_trace
    if num_workers > 0:
        worker_seeds = {record[2] for record in first_seed_trace}
        assert len(worker_seeds) == num_workers

    rank0, rank1 = distributed_sampler_entry()
    assert sorted(rank0 + rank1) == list(range(8))
    assert set(rank0).isdisjoint(rank1)
    assert "mutually exclusive" in invalid_sampler_configuration()

    degraded, message = transfer_batch(
        batch,
        request_cuda=True,
        cuda_available=False,
    )
    assert degraded.tokens.device.type == "cpu"
    assert message.startswith("CUDA unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("cpu", "auto"),
        default="cpu",
        help="auto tries CUDA; CPU remains the default reproducible smoke path",
    )
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    run_checks(num_workers=args.num_workers)
    request_cuda = args.device == "auto"
    use_pin_memory = request_cuda and torch.cuda.is_available()
    loader = make_demo_loader(args.num_workers, pin_memory=use_pin_memory)
    batch = next(iter(loader))
    moved, transfer_status = transfer_batch(batch, request_cuda=request_cuda)

    observations = compare_loader_configs(pin_memory=use_pin_memory)
    rank0, rank1 = distributed_sampler_entry()
    seed_trace = collect_seed_records(num_workers=args.num_workers)

    print("map batch ids:", batch.ids.tolist())
    print("map batch shape:", tuple(batch.tokens.shape))
    print("sampler batches (keep tail):", sampler_batches(drop_last=False))
    print("sampler batches (drop tail):", sampler_batches(drop_last=True))
    print("worker seed trace reproducible:", seed_trace == collect_seed_records(args.num_workers))
    print("DistributedSampler rank indices:", rank0, rank1)
    print("transfer device:", moved.tokens.device.type)
    print("transfer status:", transfer_status)
    print("local loader observations (fixed work; no speed threshold):")
    for observation in observations:
        print(observation)
    print("compare only under the same script, input, environment, and process policy")
    print("all dataloader-pipeline checks passed")


if __name__ == "__main__":
    main()
