"""CPU experiments for PyTorch Autograd mechanics.

Run from the repository root:
    python3 examples/pytorch/autograd_mechanics.py
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd import Function


class CustomSinh(Function):
    """A small custom operation with an explicit reverse-mode derivative."""

    @staticmethod
    def forward(input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute sinh without asking autograd to record the internal operation."""
        return input_tensor.sinh()

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """Save only the tensor needed to evaluate cosh(x) in backward."""
        del output
        (input_tensor,) = inputs
        ctx.save_for_backward(input_tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Return grad_output * d(sinh(x))/dx."""
        (input_tensor,) = ctx.saved_tensors
        return grad_output * input_tensor.cosh()


class WrongSquare(Function):
    """Intentionally wrong derivative used to prove that gradcheck can fail."""

    @staticmethod
    def forward(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.square()

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del output
        (input_tensor,) = inputs
        ctx.save_for_backward(input_tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = ctx.saved_tensors
        # 故意写错：x^2 的导数应为 2x，而不是 3x。
        return grad_output * 3 * input_tensor


def dynamic_graph_demo() -> dict[str, Any]:
    """Build two graphs selected by ordinary Python control flow."""
    square_input = torch.tensor(2.0, requires_grad=True)
    square_output = square_input.square()
    square_output.backward()

    linear_input = torch.tensor(2.0, requires_grad=True)
    linear_output = linear_input * 3
    linear_output.backward()

    return {
        "square_grad_fn": type(square_output.grad_fn).__name__,
        "linear_grad_fn": type(linear_output.grad_fn).__name__,
        "square_gradient": square_input.grad.item(),
        "linear_gradient": linear_input.grad.item(),
    }


def leaf_and_saved_tensor_demo() -> dict[str, Any]:
    """Inspect leaf state, retained non-leaf gradients, and saved tensors."""
    leaf = torch.tensor(2.0, requires_grad=True)
    non_leaf = leaf * 3
    non_leaf.retain_grad()
    loss = non_leaf.square()
    loss.backward()

    saved_input = torch.tensor([0.25, -0.5], dtype=torch.double, requires_grad=True)
    custom_output = CustomSinh.apply(saved_input)
    saved_tensors = custom_output.grad_fn.saved_tensors

    return {
        "leaf_is_leaf": leaf.is_leaf,
        "leaf_grad_fn_is_none": leaf.grad_fn is None,
        "non_leaf_is_leaf": non_leaf.is_leaf,
        "non_leaf_grad_fn": type(non_leaf.grad_fn).__name__,
        "leaf_gradient": leaf.grad.item(),
        "retained_non_leaf_gradient": non_leaf.grad.item(),
        "saved_tensor_count": len(saved_tensors),
        "saved_tensor_matches_input": torch.equal(saved_tensors[0], saved_input),
    }


def gradient_accumulation_demo() -> tuple[float, float]:
    """Show that separate backward calls add into a leaf's .grad field."""
    value = torch.tensor(2.0, requires_grad=True)
    (value * 3).backward()
    first = value.grad.item()
    (value * 4).backward()
    accumulated = value.grad.item()
    return first, accumulated


def graph_control_demo() -> dict[str, float]:
    """Exercise retain_graph for reuse and create_graph for higher derivatives."""
    retained_input = torch.tensor(3.0, requires_grad=True)
    retained_output = retained_input.square()
    retained_output.backward(retain_graph=True)
    first_backward = retained_input.grad.item()
    retained_input.grad.zero_()
    retained_output.backward()
    second_backward = retained_input.grad.item()

    higher_order_input = torch.tensor(2.0, requires_grad=True)
    cubic = higher_order_input.pow(3)
    first_derivative = torch.autograd.grad(
        cubic,
        higher_order_input,
        create_graph=True,
    )[0]
    second_derivative = torch.autograd.grad(
        first_derivative,
        higher_order_input,
    )[0]

    return {
        "first_backward": first_backward,
        "second_backward": second_backward,
        "first_derivative": first_derivative.item(),
        "second_derivative": second_derivative.item(),
    }


def mode_demo() -> dict[str, Any]:
    """Compare detach, no_grad, and inference_mode on CPU."""
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    intermediate = source * 2
    detached = intermediate.detach()

    with torch.no_grad():
        no_grad_value = source * 3
    later_input = torch.ones(2, requires_grad=True)
    (later_input * no_grad_value).sum().backward()

    with torch.inference_mode():
        inference_value = torch.ones(2)
    tracked_input = torch.ones(2, requires_grad=True)
    inference_error = ""
    try:
        (tracked_input * inference_value).sum().backward()
    except RuntimeError as error:
        inference_error = str(error)

    # clone() outside inference mode creates a normal tensor that can be saved.
    recovered = inference_value.clone()
    recovered_input = torch.ones(2, requires_grad=True)
    (recovered_input * recovered).sum().backward()

    return {
        "detached_requires_grad": detached.requires_grad,
        "detached_shares_storage": (
            detached.untyped_storage().data_ptr()
            == intermediate.untyped_storage().data_ptr()
        ),
        "no_grad_requires_grad": no_grad_value.requires_grad,
        "no_grad_reuse_gradient": later_input.grad.tolist(),
        "inference_reuse_failed": bool(inference_error),
        "inference_error": inference_error,
        "clone_reuse_gradient": recovered_input.grad.tolist(),
    }


def hook_demo() -> dict[str, Any]:
    """Observe and replace an incoming gradient with Tensor.register_hook."""
    value = torch.tensor(3.0, requires_grad=True)
    squared = value.square()
    observed: list[float] = []

    def halve_gradient(gradient: torch.Tensor) -> torch.Tensor:
        observed.append(gradient.item())
        return gradient * 0.5

    handle = squared.register_hook(halve_gradient)
    (4 * squared).backward()
    handle.remove()

    return {
        "hook_observed": observed,
        "leaf_gradient_after_hook": value.grad.item(),
    }


def inplace_version_error() -> dict[str, Any]:
    """Trigger the version-counter guard after mutating a saved tensor."""
    value = torch.tensor([2.0], requires_grad=True)
    output = value.square()
    before = value._version
    with torch.no_grad():
        value.add_(1)
    after = value._version

    message = ""
    try:
        output.backward()
    except RuntimeError as error:
        message = str(error)

    if not message:
        raise AssertionError("expected in-place version mismatch to raise RuntimeError")
    return {"version_before": before, "version_after": after, "error": message}


def custom_function_checks() -> dict[str, Any]:
    """Validate one correct and one intentionally incorrect custom derivative."""
    correct_input = torch.tensor(
        [-0.7, 0.2, 1.1],
        dtype=torch.double,
        requires_grad=True,
    )
    passed = torch.autograd.gradcheck(CustomSinh.apply, (correct_input,))

    wrong_input = torch.tensor(
        [-0.8, 0.4, 1.2],
        dtype=torch.double,
        requires_grad=True,
    )
    wrong_error = ""
    try:
        torch.autograd.gradcheck(WrongSquare.apply, (wrong_input,))
    except RuntimeError as error:
        wrong_error = str(error)

    if not wrong_error:
        raise AssertionError("gradcheck should reject WrongSquare.backward")
    return {"correct_gradcheck": passed, "wrong_gradcheck_error": wrong_error}


def run_checks() -> None:
    """Assert every invariant demonstrated by the lesson."""
    dynamic = dynamic_graph_demo()
    assert dynamic["square_grad_fn"] != dynamic["linear_grad_fn"]
    assert dynamic["square_gradient"] == 4.0
    assert dynamic["linear_gradient"] == 3.0

    leaf = leaf_and_saved_tensor_demo()
    assert leaf["leaf_is_leaf"]
    assert leaf["leaf_grad_fn_is_none"]
    assert not leaf["non_leaf_is_leaf"]
    assert leaf["leaf_gradient"] == 36.0
    assert leaf["retained_non_leaf_gradient"] == 12.0
    assert leaf["saved_tensor_count"] == 1
    assert leaf["saved_tensor_matches_input"]

    assert gradient_accumulation_demo() == (3.0, 7.0)

    graph = graph_control_demo()
    assert graph == {
        "first_backward": 6.0,
        "second_backward": 6.0,
        "first_derivative": 12.0,
        "second_derivative": 12.0,
    }

    modes = mode_demo()
    assert not modes["detached_requires_grad"]
    assert modes["detached_shares_storage"]
    assert not modes["no_grad_requires_grad"]
    assert modes["no_grad_reuse_gradient"] == [3.0, 6.0]
    assert modes["inference_reuse_failed"]
    assert modes["clone_reuse_gradient"] == [1.0, 1.0]

    hook = hook_demo()
    assert hook["hook_observed"] == [4.0]
    assert hook["leaf_gradient_after_hook"] == 12.0

    version = inplace_version_error()
    assert version["version_after"] == version["version_before"] + 1
    assert "inplace" in version["error"].lower()

    custom = custom_function_checks()
    assert custom["correct_gradcheck"]
    assert custom["wrong_gradcheck_error"]


def main() -> None:
    run_checks()

    dynamic = dynamic_graph_demo()
    accumulated = gradient_accumulation_demo()
    graph = graph_control_demo()
    modes = mode_demo()
    hook = hook_demo()
    version = inplace_version_error()
    custom = custom_function_checks()

    print("dynamic grad_fn:", dynamic["square_grad_fn"], "/", dynamic["linear_grad_fn"])
    print("accumulated gradients:", accumulated)
    print("first/second derivative:", graph["first_derivative"], graph["second_derivative"])
    print("detach shares storage:", modes["detached_shares_storage"])
    print("inference tensor reuse rejected:", modes["inference_reuse_failed"])
    print("hook observed / leaf grad:", hook["hook_observed"], hook["leaf_gradient_after_hook"])
    print("version counter:", version["version_before"], "->", version["version_after"])
    print("CustomSinh gradcheck:", custom["correct_gradcheck"])
    print("WrongSquare gradcheck rejected:", bool(custom["wrong_gradcheck_error"]))
    print("all autograd-mechanics checks passed")


if __name__ == "__main__":
    main()
