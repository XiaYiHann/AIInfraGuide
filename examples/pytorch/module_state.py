"""CPU experiments for ``nn.Module`` registration and model state.

Run from the repository root:
    python3 examples/pytorch/module_state.py
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ModuleStateDemo(nn.Module):
    """A small module containing every state category used by the lesson."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)
        self.scale = nn.Parameter(torch.tensor(2.0))

        # Both are buffers and follow module-wide conversions, but only the
        # persistent one belongs to state_dict.
        self.register_buffer("running_total", torch.zeros(3), persistent=True)
        self.register_buffer("scratch", torch.ones(3), persistent=False)

        # A plain Tensor is just a Python attribute: Module does not manage it.
        self.plain_tensor = torch.full((3,), -1.0)

        # These containers register their contents.
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False)])
        self.scales = nn.ParameterList([torch.tensor(0.5)])

        # These plain Python lists intentionally demonstrate the failure path.
        self.unregistered_layers = [nn.Linear(3, 3, bias=False)]
        self.unregistered_parameters = [nn.Parameter(torch.tensor(7.0))]
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(value)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.dropout(hidden * self.scale * self.scales[0])


class SharedParameterDemo(nn.Module):
    """Two attribute paths intentionally point to one Parameter object."""

    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.tensor([1.0, -1.0]))
        self.encoder_scale = shared
        self.decoder_scale = shared


def initialize_linears(module: nn.Module) -> None:
    """Deterministically initialize every registered Linear child."""
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0.25)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def frozen_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a state_dict so later in-place updates cannot mutate the snapshot."""
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def registration_demo() -> dict[str, Any]:
    """Report which attributes Module registration makes visible."""
    model = ModuleStateDemo()
    parameter_names = [name for name, _ in model.named_parameters()]
    buffer_names = [name for name, _ in model.named_buffers()]
    state_keys = list(model.state_dict())

    return {
        "parameter_names": parameter_names,
        "buffer_names": buffer_names,
        "state_keys": state_keys,
        "module_list_registered": "layers.0.weight" in parameter_names,
        "parameter_list_registered": "scales.0" in parameter_names,
        "python_module_list_missing": not any(
            name.startswith("unregistered_layers") for name in parameter_names
        ),
        "python_parameter_list_missing": not any(
            name.startswith("unregistered_parameters") for name in parameter_names
        ),
        "persistent_buffer_saved": "running_total" in state_keys,
        "temporary_buffer_not_saved": "scratch" not in state_keys,
        "plain_tensor_not_saved": "plain_tensor" not in state_keys,
    }


def conversion_demo() -> dict[str, torch.dtype]:
    """Show that registered state follows ``Module.to`` and plain values do not."""
    model = ModuleStateDemo()
    model.to(dtype=torch.float64)
    return {
        "parameter": model.scale.dtype,
        "persistent_buffer": model.running_total.dtype,
        "temporary_buffer": model.scratch.dtype,
        "plain_tensor": model.plain_tensor.dtype,
        "python_list_module": model.unregistered_layers[0].weight.dtype,
        "python_list_parameter": model.unregistered_parameters[0].dtype,
    }


def shallow_reference_demo() -> dict[str, Any]:
    """Prove that state_dict is a shallow mapping, not a frozen checkpoint."""
    model = ModuleStateDemo()
    live_state = model.state_dict()
    before = live_state["scale"].clone()
    same_storage = live_state["scale"].data_ptr() == model.scale.data_ptr()
    with torch.no_grad():
        model.scale.add_(3.0)
    return {
        "same_storage": same_storage,
        "before": before.item(),
        "after": live_state["scale"].item(),
    }


def round_trip_demo() -> dict[str, Any]:
    """Restore parameters and persistent buffers from an independent snapshot."""
    model = ModuleStateDemo()
    model.apply(initialize_linears)
    with torch.no_grad():
        model.scale.fill_(3.0)
        model.running_total.copy_(torch.tensor([1.0, 2.0, 3.0]))
        model.scratch.fill_(4.0)
        model.plain_tensor.fill_(5.0)

    snapshot = frozen_state_dict(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
        model.running_total.add_(10.0)
        model.scratch.add_(10.0)
        model.plain_tensor.add_(10.0)

    incompatible = model.load_state_dict(snapshot)
    return {
        "scale": model.scale.item(),
        "running_total": model.running_total.tolist(),
        "scratch": model.scratch.tolist(),
        "plain_tensor": model.plain_tensor.tolist(),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }


def strict_load_demo() -> dict[str, Any]:
    """Exercise both diagnosable non-strict loading and strict rejection."""
    model = ModuleStateDemo()
    partial = frozen_state_dict(model)
    partial.pop("projection.weight")
    partial["legacy.weight"] = torch.ones(1)

    incompatible = model.load_state_dict(partial, strict=False)
    strict_error = ""
    try:
        model.load_state_dict(partial, strict=True)
    except RuntimeError as error:
        strict_error = str(error)

    if not strict_error:
        raise AssertionError("strict=True should reject mismatched state_dict keys")
    return {
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "strict_error": strict_error,
    }


def sharing_demo() -> dict[str, Any]:
    """Show deduplication during traversal and sharing after state loading."""
    model = SharedParameterDemo()
    state_keys = list(model.state_dict())
    named_parameters = [name for name, _ in model.named_parameters()]
    snapshot = frozen_state_dict(model)

    with torch.no_grad():
        model.encoder_scale.zero_()
    model.load_state_dict(snapshot)

    return {
        "same_object": model.encoder_scale is model.decoder_scale,
        "same_storage": (
            model.encoder_scale.data_ptr() == model.decoder_scale.data_ptr()
        ),
        "state_keys": state_keys,
        "named_parameters": named_parameters,
        "restored_value": model.encoder_scale.detach().tolist(),
    }


def mode_and_hook_demo() -> dict[str, Any]:
    """Separate train/eval mode from grad mode and expose Module hook entry points."""
    model = ModuleStateDemo()
    model.train()
    train_flags = [module.training for module in model.modules()]
    model.eval()
    eval_flags = [module.training for module in model.modules()]

    events: list[str] = []

    def before_forward(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        del module, inputs
        events.append("forward_pre")

    def after_forward(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs, output
        events.append("forward")

    pre_handle = model.register_forward_pre_hook(before_forward)
    post_handle = model.register_forward_hook(after_forward)
    value = torch.ones(1, 3, requires_grad=True)
    output = model(value)
    pre_handle.remove()
    post_handle.remove()

    return {
        "all_training_before": all(train_flags),
        "all_eval_after": not any(eval_flags),
        "output_requires_grad_in_eval": output.requires_grad,
        "parameter_still_requires_grad": model.scale.requires_grad,
        "events": events,
    }


def run_checks() -> None:
    """Assert every invariant demonstrated by the lesson."""
    registration = registration_demo()
    assert registration["module_list_registered"]
    assert registration["parameter_list_registered"]
    assert registration["python_module_list_missing"]
    assert registration["python_parameter_list_missing"]
    assert registration["persistent_buffer_saved"]
    assert registration["temporary_buffer_not_saved"]
    assert registration["plain_tensor_not_saved"]

    conversion = conversion_demo()
    assert conversion["parameter"] == torch.float64
    assert conversion["persistent_buffer"] == torch.float64
    assert conversion["temporary_buffer"] == torch.float64
    assert conversion["plain_tensor"] == torch.float32
    assert conversion["python_list_module"] == torch.float32
    assert conversion["python_list_parameter"] == torch.float32

    shallow = shallow_reference_demo()
    assert shallow == {"same_storage": True, "before": 2.0, "after": 5.0}

    round_trip = round_trip_demo()
    assert round_trip["scale"] == 3.0
    assert round_trip["running_total"] == [1.0, 2.0, 3.0]
    assert round_trip["scratch"] == [14.0, 14.0, 14.0]
    assert round_trip["plain_tensor"] == [15.0, 15.0, 15.0]
    assert not round_trip["missing_keys"]
    assert not round_trip["unexpected_keys"]

    strict = strict_load_demo()
    assert strict["missing_keys"] == ["projection.weight"]
    assert strict["unexpected_keys"] == ["legacy.weight"]
    assert "Missing key(s)" in strict["strict_error"]
    assert "Unexpected key(s)" in strict["strict_error"]

    sharing = sharing_demo()
    assert sharing["same_object"]
    assert sharing["same_storage"]
    assert sharing["state_keys"] == ["encoder_scale", "decoder_scale"]
    assert sharing["named_parameters"] == ["encoder_scale"]
    assert sharing["restored_value"] == [1.0, -1.0]

    modes = mode_and_hook_demo()
    assert modes["all_training_before"]
    assert modes["all_eval_after"]
    assert modes["output_requires_grad_in_eval"]
    assert modes["parameter_still_requires_grad"]
    assert modes["events"] == ["forward_pre", "forward"]


def main() -> None:
    run_checks()
    registration = registration_demo()
    strict = strict_load_demo()
    sharing = sharing_demo()
    modes = mode_and_hook_demo()

    print("state_dict keys:", registration["state_keys"])
    print("registered buffers:", registration["buffer_names"])
    print("python-list contents omitted:", registration["python_module_list_missing"])
    print("strict=False missing keys:", strict["missing_keys"])
    print("strict=False unexpected keys:", strict["unexpected_keys"])
    print("shared parameter kept:", sharing["same_object"])
    print("module hook events:", modes["events"])
    print("all module-state checks passed")


if __name__ == "__main__":
    main()
