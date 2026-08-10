"""MiniGPT capstone: a tiny, observable character-level causal Transformer.

Run from the repository root:
    python3 examples/pytorch/minigpt_project.py smoke --device cpu \
        --output-dir /tmp/aiinfraguide-pytorch-4.12
    python3 examples/pytorch/minigpt_project.py ddp \
        --output-dir /tmp/aiinfraguide-pytorch-4.12-ddp

The script emits correctness and resource-observation evidence as JSON. It does
not download a model or dataset, benchmark speed, or require dependencies beyond
PyTorch and the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, record_function
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


TEACHING_CORPUS = (
    "ai infra turns models into reliable systems.\n"
    "measure first, optimize second, and keep evidence.\n"
    "small tests make state, data, and failures visible.\n"
) * 8
CHECKPOINT_SCHEMA_VERSION = "1.0"
EVIDENCE_SCHEMA_VERSION = "1.0"
DDP_WORLD_SIZE = 2


@dataclass(frozen=True)
class Config:
    """Model dimensions and bounded training/profiling budgets."""

    seed: int = 20260810
    vocab_size: int = 0
    block_size: int = 8
    batch_size: int = 4
    d_model: int = 16
    n_heads: int = 2
    n_layers: int = 1
    ff_multiplier: int = 2
    dropout: float = 0.0
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    smoke_steps: int = 2
    profiler_wait: int = 1
    profiler_warmup: int = 1
    profiler_active: int = 1
    profiler_repeat: int = 1
    ddp_dataset_size: int = 16

    def validate(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.block_size <= 1 or self.batch_size <= 0:
            raise ValueError("block_size must exceed one and batch_size must be positive")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers <= 0 or self.ff_multiplier <= 0:
            raise ValueError("n_layers and ff_multiplier must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.smoke_steps <= 0:
            raise ValueError("smoke_steps must be positive")
        if self.profiler_wait < 0 or self.profiler_warmup < 0:
            raise ValueError("profiler wait and warmup must be non-negative")
        if self.profiler_active <= 0:
            raise ValueError("profiler active must be positive")
        if self.profiler_repeat != 1:
            raise ValueError(
                "the MiniGPT single-trace exporter requires profiler_repeat=1"
            )
        if self.ddp_dataset_size <= 0 or self.ddp_dataset_size % DDP_WORLD_SIZE:
            raise ValueError("ddp_dataset_size must be positive and divisible by two")


@dataclass(frozen=True)
class AmpPolicy:
    device_type: str
    enabled: bool
    autocast_dtype: str
    reason: str


class CharacterVocabulary:
    """A deterministic character vocabulary sorted by Unicode code point."""

    def __init__(self, characters: Sequence[str]) -> None:
        unique = tuple(characters)
        if len(unique) != len(set(unique)) or not unique:
            raise ValueError("vocabulary characters must be non-empty and unique")
        self.characters = unique
        self.stoi = {character: index for index, character in enumerate(unique)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterVocabulary":
        if not text:
            raise ValueError("cannot build a vocabulary from empty text")
        return cls(sorted(set(text)))

    def encode(self, text: str) -> torch.Tensor:
        try:
            values = [self.stoi[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character is absent from vocabulary: {error.args[0]!r}") from error
        return torch.tensor(values, dtype=torch.long)

    def decode(self, token_ids: Iterable[int]) -> str:
        decoded: list[str] = []
        for raw_index in token_ids:
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValueError("token id is outside the vocabulary") from error
            if not 0 <= index < len(self.characters):
                raise ValueError("token id is outside the vocabulary")
            decoded.append(self.characters[index])
        return "".join(decoded)


class SlidingWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return x[t:t+T] and the one-token-shifted causal target."""

    def __init__(self, token_ids: torch.Tensor, block_size: int) -> None:
        if token_ids.ndim != 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a one-dimensional torch.long tensor")
        if block_size <= 1 or token_ids.numel() <= block_size:
            raise ValueError("block_size must exceed one and leave at least one window")
        self.token_ids = token_ids.detach().clone()
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return self.token_ids.numel() - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        inputs = self.token_ids[index : index + self.block_size]
        targets = self.token_ids[index + 1 : index + self.block_size + 1]
        return inputs.clone(), targets.clone()


class CausalBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(config.d_model)
        hidden = config.d_model * config.ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.ffn(self.norm_ffn(hidden))


class MiniGPT(nn.Module):
    """A small decoder-only causal Transformer suitable for CPU smoke tests."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @staticmethod
    def causal_mask(sequence_length: int, device: torch.device | str = "cpu") -> torch.Tensor:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        return torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        batch_size, sequence_length = token_ids.shape
        del batch_size
        if sequence_length > self.config.block_size:
            raise ValueError("sequence length exceeds configured block_size")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        mask = self.causal_mask(sequence_length, token_ids.device)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.lm_head(self.final_norm(hidden))


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_project(config: Config | None = None) -> tuple[Config, CharacterVocabulary, SlidingWindowDataset]:
    vocabulary = CharacterVocabulary.from_text(TEACHING_CORPUS)
    resolved = replace(config or Config(), vocab_size=len(vocabulary.characters))
    resolved.validate()
    dataset = SlidingWindowDataset(vocabulary.encode(TEACHING_CORPUS), resolved.block_size)
    return resolved, vocabulary, dataset


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device request: {requested}")


def make_amp_policy(device: torch.device) -> AmpPolicy:
    if device.type == "cuda":
        return AmpPolicy(
            device_type="cuda",
            enabled=True,
            autocast_dtype="float16",
            reason="CUDA path uses torch.autocast(float16) with GradScaler",
        )
    return AmpPolicy(
        device_type="cpu",
        enabled=False,
        autocast_dtype="float32",
        reason="CPU correctness path explicitly disables autocast and uses float32",
    )


def make_grad_scaler(policy: AmpPolicy) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=policy.enabled)


def _autocast_context(policy: AmpPolicy):
    if not policy.enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def flatten_parameters(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().to("cpu").reshape(-1) for parameter in module.parameters()])


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: tuple[torch.Tensor, torch.Tensor],
    config: Config,
    device: torch.device,
    policy: AmpPolicy,
) -> dict[str, Any]:
    """Run one update and return correctness observations, never speed claims."""
    model.train()
    inputs, targets = (tensor.to(device) for tensor in batch)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)

    with record_function("minigpt.train_step"):
        with record_function("minigpt.forward"):
            with _autocast_context(policy):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                )
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError("training loss is not finite")
        with record_function("minigpt.backward"):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.max_grad_norm,
                error_if_nonfinite=True,
            )
        with record_function("minigpt.optimizer"):
            scaler.step(optimizer)
            scaler.update()

    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )
    if not changed:
        raise AssertionError("optimizer step did not update any parameter")
    return {
        "loss": float(loss.detach().cpu()),
        "finite_loss": True,
        "parameter_updated": True,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "logits_shape": list(logits.shape),
    }


def capture_rng_state(include_cuda: bool) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def required_checkpoint_fields() -> set[str]:
    return {
        "schema_version",
        "model",
        "optimizer",
        "scaler",
        "config",
        "step",
        "rng",
        "vocab",
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Config,
    step: int,
    vocabulary: CharacterVocabulary,
    device: torch.device,
) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "step": int(step),
        "rng": capture_rng_state(include_cuda=device.type == "cuda"),
        "vocab": list(vocabulary.characters),
    }

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Config,
    vocabulary: CharacterVocabulary,
    device: torch.device,
) -> int:
    # Deserialize portable state on CPU first. In particular, CPU and CUDA RNG
    # state APIs both require CPU ByteTensors; mapping the whole payload to CUDA
    # would make checkpoint restore fail before model state can be copied.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    missing = required_checkpoint_fields() - set(payload)
    if missing:
        raise RuntimeError("checkpoint is missing fields: " + ", ".join(sorted(missing)))
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint schema version")
    if payload["config"] != asdict(config):
        raise RuntimeError("checkpoint Config does not match the requested project Config")
    if tuple(payload["vocab"]) != vocabulary.characters:
        raise RuntimeError("checkpoint vocabulary does not match the project vocabulary")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"])
    return int(payload["step"])


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def memory_report(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    policy: AmpPolicy,
) -> dict[str, Any]:
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in model.parameters())
    gradient_bytes = sum(
        _tensor_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer_bytes = _tensor_bytes(optimizer.state)
    assumed_element_bytes = 2 if policy.enabled else 4
    activation_elements = config.batch_size * config.block_size * (
        config.d_model * (config.n_layers + 2) + config.vocab_size
    )
    activation_estimate = activation_elements * assumed_element_bytes
    temporary_estimate = config.block_size * config.block_size  # boolean causal mask

    cuda_available = torch.cuda.is_available()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_observation: dict[str, Any] = {
            "available": True,
            "measured": True,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "note": "current-run allocator observations; not a benchmark",
        }
    else:
        cuda_observation = {
            "available": cuda_available,
            "measured": False,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "note": "CPU mode selected; GPU memory behavior was not measured",
        }

    report = {
        "scope": "correctness_and_resource_observation_not_performance",
        "logical_bytes": {
            "parameters": int(parameter_bytes),
            "gradients": int(gradient_bytes),
            "optimizer_state": int(optimizer_bytes),
            "activations_estimate": int(activation_estimate),
            "temporary_buffers_estimate": int(temporary_estimate),
        },
        "activation_estimate": {
            "bytes": int(activation_estimate),
            "is_estimate": True,
            "assumed_bytes_per_element": assumed_element_bytes,
            "method": "batch*sequence*(embedding+block outputs+logits); excludes backend workspaces",
        },
        "cuda": cuda_observation,
    }
    validate_memory_report(report)
    return report


def validate_memory_report(report: Mapping[str, Any]) -> None:
    required_logical = {
        "parameters",
        "gradients",
        "optimizer_state",
        "activations_estimate",
        "temporary_buffers_estimate",
    }
    if set(report.get("logical_bytes", {})) != required_logical:
        raise ValueError("memory report logical_bytes schema is incomplete")
    if any(int(value) < 0 for value in report["logical_bytes"].values()):
        raise ValueError("memory report byte counts must be non-negative")
    cuda = report.get("cuda", {})
    required_cuda = {
        "available",
        "measured",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "note",
    }
    if set(cuda) != required_cuda:
        raise ValueError("memory report CUDA schema is incomplete")
    if cuda["measured"]:
        if cuda["allocated_bytes"] > cuda["reserved_bytes"]:
            raise ValueError("CUDA allocated bytes cannot exceed reserved bytes")
        if cuda["peak_allocated_bytes"] < cuda["allocated_bytes"]:
            raise ValueError("CUDA peak allocated bytes cannot be below current allocated")


def validate_trace(path: str | Path, required_marker: str = "minigpt.train_step") -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise RuntimeError("profiler trace has no traceEvents")
    names = {event.get("name") for event in events if isinstance(event, dict)}
    if required_marker not in names:
        raise RuntimeError(f"profiler trace is missing marker: {required_marker}")
    return {"trace_events": len(events), "trace_bytes": Path(path).stat().st_size}


def run_profiler(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    config: Config,
    device: torch.device,
    policy: AmpPolicy,
    trace_path: str | Path,
) -> dict[str, Any]:
    destination = Path(trace_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    schedule = torch.profiler.schedule(
        wait=config.profiler_wait,
        warmup=config.profiler_warmup,
        active=config.profiler_active,
        repeat=config.profiler_repeat,
    )
    total_steps = (
        config.profiler_wait + config.profiler_warmup + config.profiler_active
    ) * config.profiler_repeat
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    iterator = iter(loader)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(destination)),
        acc_events=True,
    ) as profiler:
        for _ in range(total_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            train_one_step(model, optimizer, scaler, batch, config, device, policy)
            profiler.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trace = validate_trace(destination)
    return {
        "activities": [activity.name for activity in activities],
        "schedule": {
            "wait": config.profiler_wait,
            "warmup": config.profiler_warmup,
            "active": config.profiler_active,
            "repeat": config.profiler_repeat,
            "steps": total_steps,
        },
        "trace": str(destination),
        **trace,
        "performance_measured": False,
    }


def _make_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _first_batch(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], config: Config):
    generator = torch.Generator().manual_seed(config.seed)
    return next(
        iter(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
            )
        )
    )


def _state_dicts_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name].detach().cpu(), right_state[name].detach().cpu())
        for name in left_state
    )


def run_serial_smoke(
    output_dir: str | Path,
    requested_device: str = "cpu",
    config: Config | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved, vocabulary, dataset = build_project(config)
    set_determinism(resolved.seed)
    device = resolve_device(requested_device)
    policy = make_amp_policy(device)
    model = MiniGPT(resolved).to(device)
    optimizer = _make_optimizer(model, resolved)
    scaler = make_grad_scaler(policy)
    batch = _first_batch(dataset, resolved)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    observations = [
        train_one_step(model, optimizer, scaler, batch, resolved, device, policy)
        for _ in range(resolved.smoke_steps)
    ]

    checkpoint_path = destination / "minigpt_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scaler,
        resolved,
        resolved.smoke_steps,
        vocabulary,
        device,
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(4)
    random.random()
    torch.rand(4)

    resumed_model = MiniGPT(resolved).to(device)
    resumed_optimizer = _make_optimizer(resumed_model, resolved)
    resumed_scaler = make_grad_scaler(policy)
    restored_step = load_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_scaler,
        resolved,
        vocabulary,
        device,
    )
    if restored_step != resolved.smoke_steps or not _state_dicts_equal(model, resumed_model):
        raise AssertionError("checkpoint did not restore model and step exactly")
    if random.random() != expected_python_random:
        raise AssertionError("checkpoint did not restore Python RNG state")
    if not torch.equal(torch.rand(4), expected_torch_random):
        raise AssertionError("checkpoint did not restore PyTorch CPU RNG state")
    resumed_observation = train_one_step(
        resumed_model,
        resumed_optimizer,
        resumed_scaler,
        batch,
        resolved,
        device,
        policy,
    )

    profiler_evidence = run_profiler(
        model,
        optimizer,
        scaler,
        dataset,
        resolved,
        device,
        policy,
        destination / "minigpt_trace.json",
    )
    resources = memory_report(model, optimizer, resolved, device, policy)
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "scope": "current_run_correctness_and_resource_observation_not_performance",
        "mode": "serial_smoke",
        "environment": {
            "pytorch": torch.__version__,
            "requested_device": requested_device,
            "selected_device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "config": asdict(resolved),
        "vocabulary": {
            "size": len(vocabulary.characters),
            "characters": list(vocabulary.characters),
            "deterministic_sorted": True,
        },
        "data": {
            "dataset_windows": len(dataset),
            "target_shift": 1,
            "downloads": False,
        },
        "amp": asdict(policy),
        "training": {
            "steps": observations,
            "all_losses_finite": all(item["finite_loss"] for item in observations),
            "all_steps_updated_parameters": all(
                item["parameter_updated"] for item in observations
            ),
            "performance_thresholds": None,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "required_fields": sorted(required_checkpoint_fields()),
            "restored_step": restored_step,
            "resumed_to_step": restored_step + 1,
            "model_exact_after_load": True,
            "rng_restored": True,
            "resume_observation": resumed_observation,
        },
        "profiler": profiler_evidence,
        "memory": resources,
        "unmeasured": (
            []
            if device.type == "cuda"
            else [
                "CUDA autocast and GradScaler enabled path",
                "CUDA allocated/reserved/peak memory",
                "GPU profiler activity",
                "GPU performance",
            ]
        ),
    }
    evidence_path = destination / "minigpt_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evidence["evidence_path"] = str(evidence_path)
    return evidence


class GlooUnavailableError(RuntimeError):
    pass


def require_gloo() -> None:
    if not dist.is_available():
        raise GlooUnavailableError("SKIP: torch.distributed is unavailable")
    if not dist.is_gloo_available():
        raise GlooUnavailableError("SKIP: Gloo is unavailable in this PyTorch build")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _ddp_worker(
    rank: int,
    world_size: int,
    rendezvous_file: str,
    output_directory: str,
    config_payload: Mapping[str, Any],
) -> None:
    destination = Path(output_directory)
    config = Config(**dict(config_payload))
    report: dict[str, Any] = {"rank": rank, "backend": "gloo"}
    failure: BaseException | None = None
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(rendezvous_file).resolve().as_uri(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )
        _, vocabulary, full_dataset = build_project(config)
        dataset = Subset(full_dataset, range(config.ddp_dataset_size))
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        sampler.set_epoch(0)
        local_indices = list(iter(sampler))
        loader = DataLoader(
            dataset,
            batch_size=config.ddp_dataset_size // world_size,
            sampler=sampler,
        )
        batch = next(iter(loader))

        set_determinism(config.seed + rank)
        model = MiniGPT(config)
        ddp_model = DDP(model)
        optimizer = _make_optimizer(ddp_model, config)
        policy = make_amp_policy(torch.device("cpu"))
        scaler = make_grad_scaler(policy)
        observation = train_one_step(
            ddp_model,
            optimizer,
            scaler,
            batch,
            config,
            torch.device("cpu"),
            policy,
        )

        local_parameters = flatten_parameters(ddp_model.module)
        gathered_parameters = [torch.empty_like(local_parameters) for _ in range(world_size)]
        dist.all_gather(gathered_parameters, local_parameters)
        local_index_tensor = torch.tensor(local_indices, dtype=torch.long)
        gathered_indices = [torch.empty_like(local_index_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_indices, local_index_tensor)
        parameter_max_diff = max(
            float((gathered_parameters[0] - candidate).abs().max())
            for candidate in gathered_parameters[1:]
        )

        checkpoint_path = destination / "minigpt_ddp_checkpoint.pt"
        if rank == 0:
            save_checkpoint(
                checkpoint_path,
                ddp_model.module,
                optimizer,
                scaler,
                config,
                step=1,
                vocabulary=vocabulary,
                device=torch.device("cpu"),
            )
        dist.barrier()
        report.update(
            {
                "world_size": world_size,
                "sampler_epoch": 0,
                "local_indices": local_indices,
                "all_rank_indices": [item.tolist() for item in gathered_indices],
                "parameter_max_diff": parameter_max_diff,
                "checkpoint_files_seen": (
                    [checkpoint_path.name] if checkpoint_path.is_file() else []
                ),
                "training": observation,
            }
        )
    except BaseException as error:
        failure = error
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        report["process_group_destroyed"] = not (
            dist.is_available() and dist.is_initialized()
        )
        _write_json(destination / f"rank_{rank}_minigpt_report.json", report)
    if failure is not None:
        raise RuntimeError(f"rank {rank} failed: {failure}") from failure


def run_ddp_smoke(
    output_dir: str | Path,
    config: Config | None = None,
) -> dict[str, Any]:
    require_gloo()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved, _, _ = build_project(config)
    for stale in destination.glob("rank_*_minigpt_report.json"):
        stale.unlink()
    checkpoint_path = destination / "minigpt_ddp_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    rendezvous_file = destination / "minigpt_gloo_rendezvous"
    if rendezvous_file.exists():
        rendezvous_file.unlink()

    try:
        mp.spawn(
            _ddp_worker,
            args=(
                DDP_WORLD_SIZE,
                str(rendezvous_file),
                str(destination),
                asdict(resolved),
            ),
            nprocs=DDP_WORLD_SIZE,
            join=True,
        )
    except Exception as error:
        raise RuntimeError(
            "two-process CPU/Gloo MiniGPT smoke failed; verify process spawning, "
            f"loopback communication, and temporary-file writes: {type(error).__name__}: {error}"
        ) from error

    reports = [
        json.loads(
            (destination / f"rank_{rank}_minigpt_report.json").read_text(encoding="utf-8")
        )
        for rank in range(DDP_WORLD_SIZE)
    ]
    failures = [report["error"] for report in reports if "error" in report]
    if failures:
        raise RuntimeError("DDP worker failure: " + " | ".join(failures))
    rank_indices = [list(map(int, report["local_indices"])) for report in reports]
    flattened = [index for shard in rank_indices for index in shard]
    disjoint = set(rank_indices[0]).isdisjoint(rank_indices[1])
    covers_dataset = sorted(flattened) == list(range(resolved.ddp_dataset_size))
    parameter_max_diff = max(float(report["parameter_max_diff"]) for report in reports)
    checkpoint_files = [checkpoint_path] if checkpoint_path.is_file() else []
    destroyed_ranks = [
        int(report["rank"])
        for report in reports
        if report["process_group_destroyed"]
    ]

    if not disjoint or not covers_dataset:
        raise AssertionError(f"DistributedSampler partition is invalid: {rank_indices}")
    if parameter_max_diff != 0.0:
        raise AssertionError(f"DDP parameters diverged: {parameter_max_diff}")
    if len(checkpoint_files) != 1:
        raise AssertionError("rank 0 must write exactly one DDP checkpoint")
    if destroyed_ranks != [0, 1]:
        raise AssertionError(f"process groups were not cleaned up: {destroyed_ranks}")
    checkpoint = torch.load(checkpoint_files[0], map_location="cpu", weights_only=True)
    if required_checkpoint_fields() - set(checkpoint):
        raise AssertionError("DDP checkpoint is missing required component-state fields")

    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "scope": "correctness_only_no_distributed_performance_measurement",
        "mode": "ddp_smoke",
        "backend": "gloo",
        "world_size": DDP_WORLD_SIZE,
        "sampler_set_epoch": True,
        "rank_indices": rank_indices,
        "shards_disjoint": disjoint,
        "shards_cover_dataset": covers_dataset,
        "parameter_max_diff": parameter_max_diff,
        "rank0_checkpoint_files": [path.name for path in checkpoint_files],
        "checkpoint_complete_fields": sorted(required_checkpoint_fields()),
        "destroyed_ranks": destroyed_ranks,
        "performance_measured": False,
        "unmeasured": [
            "CUDA/NCCL execution",
            "multi-node behavior",
            "DDP throughput or scaling efficiency",
            "FSDP and distributed checkpoint sharding",
        ],
    }
    evidence_path = destination / "minigpt_ddp_evidence.json"
    _write_json(evidence_path, evidence)
    evidence["evidence_path"] = str(evidence_path)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "ddp"), nargs="?", default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/aiinfraguide-pytorch-4.12"),
    )
    arguments = parser.parse_args()

    try:
        if arguments.mode == "ddp":
            if arguments.device != "cpu":
                raise ValueError("the teaching DDP smoke is CPU/Gloo only; use --device cpu")
            evidence = run_ddp_smoke(arguments.output_dir)
        else:
            evidence = run_serial_smoke(arguments.output_dir, arguments.device)
    except GlooUnavailableError as error:
        print(error)
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        return 1

    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    print("all MiniGPT project checks passed; no performance threshold was used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
