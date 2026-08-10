"""A deterministic, resumable PyTorch training loop for CPU smoke tests.

Run from the repository root:
    python3 examples/pytorch/training_loop.py
"""

from __future__ import annotations

import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Iterable, Sequence

import torch
from torch import nn


CHECKPOINT_VERSION = 1
REQUIRED_CHECKPOINT_KEYS = {
    "format_version",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "progress",
    "rng",
}


class TinyRegressor(nn.Module):
    """A tiny model whose optional Dropout makes RNG restoration observable."""

    def __init__(self, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Linear(8, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass
class AmpPolicy:
    """The explicit autocast/scaler policy used by one training process."""

    device: torch.device
    enabled: bool
    dtype: torch.dtype
    scaler: torch.amp.GradScaler | None
    reason: str

    def autocast(self) -> ContextManager[Any]:
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)


@dataclass
class UpdateResult:
    """Observable values from one optimizer update."""

    mean_loss: float
    pre_clip_norm: float
    post_clip_norm: float
    optimizer_ran: bool


def set_seed(seed: int) -> None:
    """Seed the Python and PyTorch generators used by this example."""
    random.seed(seed)
    torch.manual_seed(seed)


def make_amp_policy(
    device: torch.device | str,
    requested: bool,
    dtype: torch.dtype | None = None,
) -> AmpPolicy:
    """Use FP32 on CPU; use CUDA autocast and scale only CUDA FP16."""
    resolved = torch.device(device)
    if not requested:
        return AmpPolicy(resolved, False, torch.float32, None, "AMP not requested")
    if resolved.type != "cuda":
        return AmpPolicy(
            resolved,
            False,
            torch.float32,
            None,
            "CPU fallback: AMP disabled; correctness runs in float32",
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA AMP was requested, but CUDA is not available")

    selected = dtype
    if selected is None:
        selected = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    if selected not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA AMP dtype must be torch.float16 or torch.bfloat16")

    # Loss scaling is needed for the CUDA FP16 path. BF16 has FP32's exponent
    # width, so this example deliberately does not attach a GradScaler to BF16.
    scaler = torch.amp.GradScaler("cuda", enabled=(selected == torch.float16))
    return AmpPolicy(
        resolved,
        True,
        selected,
        scaler if scaler.is_enabled() else None,
        f"CUDA autocast enabled with {selected}",
    )


def make_regression_batches(
    num_batches: int = 12,
    batch_size: int = 8,
    seed: int = 17,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create fixed in-memory batches; no dataset download is needed."""
    if num_batches <= 0 or batch_size <= 0:
        raise ValueError("num_batches and batch_size must both be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 2, generator=generator)
        targets = 1.75 * inputs[:, :1] - 0.8 * inputs[:, 1:] + 0.35
        batches.append((inputs, targets))
    return batches


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Compute the global L2 norm without modifying gradients."""
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(gradient, ord=2) for gradient in gradients])
    return float(torch.linalg.vector_norm(norms, ord=2).item())


def train_one_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    amp: AmpPolicy,
    max_grad_norm: float | None = None,
) -> UpdateResult:
    """Accumulate equal-sized micro-batches and perform one optimizer update."""
    if not micro_batches:
        raise ValueError("one optimizer update requires at least one micro-batch")
    batch_sizes = {int(inputs.shape[0]) for inputs, _ in micro_batches}
    if len(batch_sizes) != 1:
        raise ValueError(
            "equal weighting matches a concatenated mean only for equal-sized micro-batches"
        )
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    accumulation_steps = len(micro_batches)

    for inputs, targets in micro_batches:
        inputs = inputs.to(amp.device)
        targets = targets.to(amp.device)
        with amp.autocast():
            loss = nn.functional.mse_loss(model(inputs), targets)
            scaled_for_accumulation = loss / accumulation_steps
        loss_sum += float(loss.detach().cpu())
        if amp.scaler is None:
            scaled_for_accumulation.backward()
        else:
            amp.scaler.scale(scaled_for_accumulation).backward()

    # Scaled FP16 gradients must be unscaled before clipping or norm checks.
    if amp.scaler is not None:
        amp.scaler.unscale_(optimizer)
    pre_clip_norm = gradient_norm(model.parameters())
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )
    post_clip_norm = gradient_norm(model.parameters())

    optimizer_ran = True
    if amp.scaler is None:
        optimizer.step()
    else:
        old_scale = amp.scaler.get_scale()
        amp.scaler.step(optimizer)
        amp.scaler.update()
        # GradScaler lowers its scale when non-finite gradients skip step().
        optimizer_ran = amp.scaler.get_scale() >= old_scale

    # Update-based schedulers advance only when the optimizer really updated.
    if scheduler is not None and optimizer_ran:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return UpdateResult(
        mean_loss=loss_sum / accumulation_steps,
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=post_clip_norm,
        optimizer_ran=optimizer_ran,
    )


def evaluate(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> float:
    """Run an explicit eval lifecycle without constructing an Autograd graph."""
    if not batches:
        raise ValueError("evaluation requires at least one batch")
    resolved = torch.device(device)
    model.eval()
    losses = []
    with torch.inference_mode():
        for inputs, targets in batches:
            prediction = model(inputs.to(resolved))
            losses.append(
                float(nn.functional.mse_loss(prediction, targets.to(resolved)).cpu())
            )
    return sum(losses) / len(losses)


def capture_rng_state(include_cuda: bool) -> dict[str, Any]:
    """Capture every RNG used by this example."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [],
    }
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("cannot capture CUDA RNG state without CUDA")
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state, rejecting a CUDA checkpoint on a CPU-only host."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_progress(progress: dict[str, Any]) -> None:
    required = {"global_step", "data_position"}
    missing = required - set(progress)
    if missing:
        raise ValueError(f"checkpoint progress is missing keys: {sorted(missing)}")
    position = progress["data_position"]
    if not isinstance(position, dict):
        raise ValueError("data_position must be a dictionary")
    if position.get("micro_step_in_update") != 0:
        raise ValueError(
            "exact checkpoints are accepted only at optimizer-update boundaries"
        )
    if int(progress["global_step"]) < 0:
        raise ValueError("global_step must be non-negative")


def _validate_checkpoint(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dictionary")
    missing = REQUIRED_CHECKPOINT_KEYS - set(payload)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    if payload["format_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version: {payload['format_version']}"
        )
    if not isinstance(payload["rng"], dict):
        raise ValueError("checkpoint RNG state must be a dictionary")
    _validate_progress(payload["progress"])
    return payload


def atomic_save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    global_step: int,
    data_position: dict[str, Any],
) -> None:
    """Write a checkpoint to a sibling temp file, fsync, then atomically replace."""
    progress = {
        "global_step": int(global_step),
        "data_position": dict(data_position),
    }
    _validate_progress(progress)
    include_cuda = any(parameter.device.type == "cuda" for parameter in model.parameters())
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "progress": progress,
        "rng": capture_rng_state(include_cuda=include_cuda),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None

        # Persist the directory entry on local POSIX filesystems when supported.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    """Validate, restore all training state, and return progress metadata."""
    payload = _validate_checkpoint(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("scheduler presence does not match the checkpoint")
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("GradScaler presence does not match the checkpoint")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"])
    return payload["progress"]


def build_components(
    seed: int,
    dropout_p: float = 0.0,
) -> tuple[TinyRegressor, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    """Build model, optimizer, and update-based scheduler deterministically."""
    set_seed(seed)
    model = TinyRegressor(dropout_p=dropout_p)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.03, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.9
    )
    return model, optimizer, scheduler


def loss_decrease_demo() -> dict[str, float]:
    """Fit a fixed synthetic regression batch and report before/after loss."""
    batches = make_regression_batches(num_batches=1, batch_size=32, seed=3)
    model, optimizer, scheduler = build_components(seed=11)
    amp = make_amp_policy("cpu", requested=True)
    before = evaluate(model, batches)
    for _ in range(40):
        train_one_update(model, optimizer, scheduler, batches, amp, max_grad_norm=5.0)
    after = evaluate(model, batches)
    return {"before": before, "after": after}


def accumulation_equivalence_demo() -> float:
    """Compare two equal micro-batches with one concatenated batch."""
    inputs = torch.tensor(
        [[1.0, -1.0], [0.5, 2.0], [-1.0, 0.25], [2.0, 1.0]]
    )
    targets = torch.tensor([[0.3], [1.1], [-0.7], [2.2]])
    micro_batches = [(inputs[:2], targets[:2]), (inputs[2:], targets[2:])]
    large_batch = [(inputs, targets)]

    set_seed(23)
    accumulated = nn.Linear(2, 1)
    direct = nn.Linear(2, 1)
    direct.load_state_dict(accumulated.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=0.05)
    amp = make_amp_policy("cpu", requested=False)

    train_one_update(
        accumulated, accumulated_optimizer, None, micro_batches, amp
    )
    train_one_update(direct, direct_optimizer, None, large_batch, amp)
    differences = [
        (left - right).abs().max().item()
        for left, right in zip(accumulated.parameters(), direct.parameters())
    ]
    return max(differences)


def clipping_demo(max_norm: float = 0.2) -> dict[str, float]:
    """Construct a large gradient and prove clipping bounds its global norm."""
    set_seed(29)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.full((4, 2), 100.0)
    targets = torch.full((4, 1), -100.0)
    result = train_one_update(
        model,
        optimizer,
        None,
        [(inputs, targets)],
        make_amp_policy("cpu", requested=False),
        max_grad_norm=max_norm,
    )
    return {
        "pre_clip_norm": result.pre_clip_norm,
        "post_clip_norm": result.post_clip_norm,
        "max_norm": max_norm,
    }


def exact_resume_demo(checkpoint_path: Path | str) -> dict[str, Any]:
    """Prove that model/optimizer/scheduler/RNG/progress resume one trajectory."""
    batches = make_regression_batches(num_batches=12, batch_size=4, seed=41)
    accumulation_steps = 2
    split_update = 3
    total_updates = len(batches) // accumulation_steps
    amp = make_amp_policy("cpu", requested=True)

    uninterrupted, optimizer, scheduler = build_components(
        seed=37, dropout_p=0.25
    )
    for update in range(split_update):
        begin = update * accumulation_steps
        train_one_update(
            uninterrupted,
            optimizer,
            scheduler,
            batches[begin : begin + accumulation_steps],
            amp,
            max_grad_norm=1.0,
        )

    next_batch_index = split_update * accumulation_steps
    atomic_save_checkpoint(
        checkpoint_path,
        uninterrupted,
        optimizer,
        scheduler,
        amp.scaler,
        global_step=split_update,
        data_position={
            "epoch": 0,
            "next_batch_index": next_batch_index,
            "micro_step_in_update": 0,
        },
    )

    uninterrupted_losses = []
    for update in range(split_update, total_updates):
        begin = update * accumulation_steps
        result = train_one_update(
            uninterrupted,
            optimizer,
            scheduler,
            batches[begin : begin + accumulation_steps],
            amp,
            max_grad_norm=1.0,
        )
        uninterrupted_losses.append(result.mean_loss)
    expected_state = {
        name: value.detach().clone()
        for name, value in uninterrupted.state_dict().items()
    }

    resumed, resumed_optimizer, resumed_scheduler = build_components(
        seed=999, dropout_p=0.25
    )
    resumed_amp = make_amp_policy("cpu", requested=True)
    progress = load_checkpoint(
        checkpoint_path,
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        resumed_amp.scaler,
    )
    resumed_losses = []
    start_update = int(progress["global_step"])
    start_batch = int(progress["data_position"]["next_batch_index"])
    for update in range(start_update, total_updates):
        offset = start_batch + (update - start_update) * accumulation_steps
        result = train_one_update(
            resumed,
            resumed_optimizer,
            resumed_scheduler,
            batches[offset : offset + accumulation_steps],
            resumed_amp,
            max_grad_norm=1.0,
        )
        resumed_losses.append(result.mean_loss)

    exact_parameters = all(
        torch.equal(expected_state[name], resumed.state_dict()[name])
        for name in expected_state
    )
    return {
        "progress": progress,
        "losses_match": resumed_losses == uninterrupted_losses,
        "parameters_match": exact_parameters,
        "learning_rate_match": (
            resumed_optimizer.param_groups[0]["lr"]
            == optimizer.param_groups[0]["lr"]
        ),
    }


def run_checks() -> None:
    """Run every invariant used by the lesson and unit tests."""
    loss = loss_decrease_demo()
    assert loss["after"] < loss["before"] * 0.2

    difference = accumulation_equivalence_demo()
    assert difference < 1e-6

    clipping = clipping_demo()
    assert clipping["pre_clip_norm"] > clipping["max_norm"]
    assert clipping["post_clip_norm"] <= clipping["max_norm"] + 1e-6

    with tempfile.TemporaryDirectory() as directory:
        resume = exact_resume_demo(Path(directory) / "training.pt")
    assert resume["losses_match"]
    assert resume["parameters_match"]
    assert resume["learning_rate_match"]
    assert resume["progress"]["global_step"] == 3
    assert resume["progress"]["data_position"]["next_batch_index"] == 6


def main() -> None:
    run_checks()
    loss = loss_decrease_demo()
    clipping = clipping_demo()
    amp = make_amp_policy("cpu", requested=True)
    with tempfile.TemporaryDirectory() as directory:
        resume = exact_resume_demo(Path(directory) / "training.pt")

    print(f"loss: {loss['before']:.6f} -> {loss['after']:.6f}")
    print("accumulation max parameter difference:", accumulation_equivalence_demo())
    print(
        "gradient norm:",
        f"{clipping['pre_clip_norm']:.6f} -> {clipping['post_clip_norm']:.6f}",
    )
    print("AMP policy:", amp.reason)
    print("checkpoint exact resume:", resume)
    print("all training-loop checks passed")


if __name__ == "__main__":
    main()
