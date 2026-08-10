"""A two-process CPU/Gloo introduction to PyTorch DDP.

Run from the repository root:
    python3 examples/pytorch/distributed_intro.py

The smoke test measures correctness only. It does not benchmark distributed
performance and does not require CUDA or NCCL.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


WORLD_SIZE = 2
DATASET_SIZE = 8
REQUIRED_TORCHRUN_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


class GlooUnavailableError(RuntimeError):
    """Raised when this PyTorch build cannot run the CPU/Gloo smoke test."""


def require_gloo() -> None:
    """Fail with an actionable message when distributed or Gloo is absent."""
    if not dist.is_available():
        raise GlooUnavailableError(
            "SKIP: torch.distributed is unavailable in this PyTorch build"
        )
    if not dist.is_gloo_available():
        raise GlooUnavailableError(
            "SKIP: the Gloo backend is unavailable; a Gloo-enabled PyTorch build is required"
        )


def read_torchrun_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, int | str]:
    """Parse the process identity and rendezvous values injected by torchrun."""
    source = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_TORCHRUN_ENV if not source.get(name)]
    if missing:
        raise RuntimeError(
            "torchrun environment is incomplete; missing: " + ", ".join(missing)
        )

    try:
        rank = int(source["RANK"])
        local_rank = int(source["LOCAL_RANK"])
        world_size = int(source["WORLD_SIZE"])
        master_port = int(source["MASTER_PORT"])
    except ValueError as error:
        raise RuntimeError("torchrun rank, world-size, and port values must be integers") from error

    if world_size <= 0 or not 0 <= rank < world_size:
        raise RuntimeError("torchrun requires 0 <= RANK < WORLD_SIZE")
    if local_rank < 0:
        raise RuntimeError("torchrun LOCAL_RANK must be non-negative")
    if not 1 <= master_port <= 65535:
        raise RuntimeError("torchrun MASTER_PORT must be in [1, 65535]")

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "master_addr": source["MASTER_ADDR"],
        "master_port": master_port,
    }


def _make_dataset() -> TensorDataset:
    """Create a divisible synthetic dataset whose sample IDs expose sharding."""
    sample_ids = torch.arange(DATASET_SIZE, dtype=torch.long)
    values = sample_ids.to(torch.float32)
    features = torch.stack((values / 8.0, (values.remainder(3) - 1.0) / 3.0), dim=1)
    targets = (1.5 * features[:, :1]) - (0.75 * features[:, 1:]) + 0.2
    return TensorDataset(sample_ids, features, targets)


def _flatten_parameters(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _ddp_worker(
    rank: int,
    world_size: int,
    rendezvous_file: str,
    output_directory: str,
) -> None:
    """Run one CPU rank and leave a report only after process-group cleanup."""
    output_dir = Path(output_directory)
    report: dict[str, object] = {"rank": rank, "backend": "gloo"}
    failure: BaseException | None = None

    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(rendezvous_file).resolve().as_uri(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        report["world_size"] = dist.get_world_size()

        dataset = _make_dataset()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=20260810,
            drop_last=False,
        )
        # set_epoch must run before creating the iterator so every epoch gets a
        # shared shuffle seed while ranks still receive different slices.
        sampler.set_epoch(0)
        loader = DataLoader(dataset, batch_size=DATASET_SIZE // world_size, sampler=sampler)
        local_ids, inputs, targets = next(iter(loader))
        report["local_indices"] = [int(value) for value in local_ids.tolist()]

        # Deliberately seed ranks differently. DDP construction synchronizes
        # the module state, after which backward synchronizes gradients.
        torch.manual_seed(1000 + rank)
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        ddp_model = DDP(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)

        optimizer.zero_grad(set_to_none=True)
        predictions = ddp_model(inputs)
        local_loss_sum = nn.functional.mse_loss(
            predictions, targets, reduction="sum"
        )
        (local_loss_sum / targets.numel()).backward()
        optimizer.step()

        # Aggregate a weighted metric instead of averaging per-rank averages.
        metric_parts = torch.tensor(
            [float(local_loss_sum.detach()), float(targets.numel())],
            dtype=torch.float64,
        )
        dist.all_reduce(metric_parts, op=dist.ReduceOp.SUM)
        global_mean_loss = metric_parts[0] / metric_parts[1]
        gathered_metrics = [torch.zeros_like(global_mean_loss) for _ in range(world_size)]
        dist.all_gather(gathered_metrics, global_mean_loss)

        local_index_tensor = local_ids.to(torch.long)
        gathered_indices = [torch.empty_like(local_index_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_indices, local_index_tensor)

        flat_parameters = _flatten_parameters(ddp_model.module)
        gathered_parameters = [torch.empty_like(flat_parameters) for _ in range(world_size)]
        dist.all_gather(gathered_parameters, flat_parameters)

        parameter_max_diff = max(
            float((gathered_parameters[0] - candidate).abs().max())
            for candidate in gathered_parameters[1:]
        )
        metric_max_diff = max(
            float((gathered_metrics[0] - candidate).abs())
            for candidate in gathered_metrics[1:]
        )
        all_indices = [
            [int(value) for value in shard.tolist()] for shard in gathered_indices
        ]

        report.update(
            {
                "all_rank_indices": all_indices,
                "global_mean_loss": float(global_mean_loss),
                "all_rank_metrics": [float(value) for value in gathered_metrics],
                "metric_max_diff": metric_max_diff,
                "parameter_max_diff": parameter_max_diff,
            }
        )

        checkpoint_path = output_dir / "rank0_checkpoint.pt"
        if rank == 0:
            torch.save(
                {
                    "saved_by_rank": rank,
                    "model": ddp_model.module.state_dict(),
                },
                checkpoint_path,
            )

        # This barrier is justified because every rank inspects a file that
        # only rank 0 publishes. It is not needed around every training step.
        dist.barrier()
        report["checkpoint_files_seen"] = (
            [checkpoint_path.name] if checkpoint_path.is_file() else []
        )
    except BaseException as error:  # Preserve cleanup evidence before failing.
        failure = error
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        report["process_group_destroyed"] = not (
            dist.is_available() and dist.is_initialized()
        )
        _write_json(output_dir / f"rank_{rank}_report.json", report)

    if failure is not None:
        raise RuntimeError(f"rank {rank} failed: {failure}") from failure


def run_distributed_smoke(output_dir: str | Path) -> dict[str, object]:
    """Spawn two Gloo ranks and validate the observable DDP invariants."""
    require_gloo()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rendezvous_file = destination / "gloo_rendezvous"
    owned_paths = [
        rendezvous_file,
        destination / "rank0_checkpoint.pt",
        *(destination / f"rank_{rank}_report.json" for rank in range(WORLD_SIZE)),
    ]
    for owned_path in owned_paths:
        if owned_path.exists():
            owned_path.unlink()

    try:
        mp.spawn(
            _ddp_worker,
            args=(WORLD_SIZE, str(rendezvous_file), str(destination)),
            nprocs=WORLD_SIZE,
            join=True,
        )
    except Exception as error:
        raise RuntimeError(
            "CPU/Gloo smoke could not start or complete. Check whether this environment "
            "permits process spawning, loopback communication, and writes to the output "
            f"directory. Original error: {type(error).__name__}: {error}"
        ) from error

    reports = [
        json.loads((destination / f"rank_{rank}_report.json").read_text(encoding="utf-8"))
        for rank in range(WORLD_SIZE)
    ]
    errors = [report.get("error") for report in reports if report.get("error")]
    if errors:
        raise RuntimeError("CPU/Gloo worker failure: " + " | ".join(str(error) for error in errors))

    rank_indices = [list(map(int, report["local_indices"])) for report in reports]
    flattened_indices = [index for shard in rank_indices for index in shard]
    shards_disjoint = set(rank_indices[0]).isdisjoint(rank_indices[1])
    shards_cover_dataset = sorted(flattened_indices) == list(range(DATASET_SIZE))
    parameter_max_diff = max(float(report["parameter_max_diff"]) for report in reports)
    metric_max_diff = max(float(report["metric_max_diff"]) for report in reports)

    checkpoint_path = destination / "rank0_checkpoint.pt"
    checkpoint_files = [checkpoint_path] if checkpoint_path.is_file() else []
    if len(checkpoint_files) != 1:
        raise AssertionError("rank 0 did not publish rank0_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    destroyed_ranks = [
        int(report["rank"])
        for report in reports
        if bool(report["process_group_destroyed"])
    ]

    result: dict[str, object] = {
        "backend": "gloo",
        "world_size": WORLD_SIZE,
        "rank_indices": rank_indices,
        "shards_disjoint": shards_disjoint,
        "shards_cover_dataset": shards_cover_dataset,
        "parameter_max_diff": parameter_max_diff,
        "global_mean_loss": float(reports[0]["global_mean_loss"]),
        "metric_max_diff": metric_max_diff,
        "checkpoint_files": [path.name for path in checkpoint_files],
        "checkpoint_saved_by_rank": int(checkpoint["saved_by_rank"]),
        "destroyed_ranks": destroyed_ranks,
        "performance_measured": False,
    }

    if not shards_disjoint or not shards_cover_dataset:
        raise AssertionError(f"DistributedSampler partition is invalid: {rank_indices}")
    if parameter_max_diff != 0.0:
        raise AssertionError(f"DDP parameters diverged across ranks: {parameter_max_diff}")
    if metric_max_diff != 0.0:
        raise AssertionError(f"all_reduce metric diverged across ranks: {metric_max_diff}")
    if result["checkpoint_saved_by_rank"] != 0:
        raise AssertionError("checkpoint was not written by rank 0")
    if destroyed_ranks != [0, 1]:
        raise AssertionError(f"process-group cleanup missing for ranks: {destroyed_ranks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Preserve rank reports and the rank-0 checkpoint in this directory",
    )
    arguments = parser.parse_args()

    try:
        if arguments.output_dir is None:
            with tempfile.TemporaryDirectory(prefix="pytorch-ddp-smoke-") as directory:
                result = run_distributed_smoke(directory)
        else:
            result = run_distributed_smoke(arguments.output_dir)
    except GlooUnavailableError as error:
        print(error)
        return 0
    except Exception as error:
        print(f"FAIL: {error}")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("all distributed-intro checks passed; no performance was measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
