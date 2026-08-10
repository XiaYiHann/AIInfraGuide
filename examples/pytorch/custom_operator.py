"""CPU-only torch.library custom-operator example for PyTorch lesson 4.9.

Run from the repository root:
    python3 examples/pytorch/custom_operator.py

The example deliberately uses a simple composition of built-in PyTorch
operators so that registration semantics can be tested without compiling an
extension. In production, this computation should remain an ordinary Python
function; custom operators are for opaque foreign code or explicit subsystem
boundaries.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


NAMESPACE = "aiinfraguide_pytorch_49"
OPERATOR_NAME = "rowwise_scaled_square"
QUALIFIED_NAME = f"{NAMESPACE}::{OPERATOR_NAME}"
SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _lookup_operator() -> torch._ops.OpOverload | None:
    """Return the existing overload without registering the name twice."""
    try:
        packet = getattr(getattr(torch.ops, NAMESPACE), OPERATOR_NAME)
        return packet.default
    except (AttributeError, RuntimeError):
        return None


def _validate_input(input_tensor: Tensor) -> None:
    """Enforce the shape and dtype contract using metadata only."""
    if input_tensor.dim() != 2:
        raise ValueError("input_tensor must be 2-D with shape (rows, columns)")
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise TypeError("input_tensor must have dtype float32 or float64")


def _register_operator_once() -> None:
    """Install schema, CPU, FakeTensor, and Autograd registrations once."""
    if _lookup_operator() is not None:
        return

    required_apis = ("custom_op", "register_fake", "register_autograd")
    missing = [name for name in required_apis if not hasattr(torch.library, name)]
    if missing:
        raise RuntimeError(
            "This example requires torch.library APIs missing from this PyTorch: "
            + ", ".join(missing)
        )

    @torch.library.custom_op(
        QUALIFIED_NAME,
        mutates_args=(),
        device_types="cpu",
    )
    def rowwise_scaled_square_cpu(input_tensor: Tensor, scale: float) -> Tensor:
        """Return scale * sum(input_tensor**2, dim=1) as a fresh CPU Tensor."""
        _validate_input(input_tensor)
        return input_tensor.square().sum(dim=1) * scale

    @torch.library.register_fake(QUALIFIED_NAME)
    def rowwise_scaled_square_fake(input_tensor: Tensor, scale: float) -> Tensor:
        """Describe output metadata without reading input data."""
        del scale
        _validate_input(input_tensor)
        return input_tensor.new_empty((input_tensor.shape[0],))

    def setup_context(ctx: Any, inputs: tuple[Tensor, float], output: Tensor) -> None:
        """Save only values needed by the registered backward formula."""
        del output
        input_tensor, scale = inputs
        ctx.save_for_backward(input_tensor)
        ctx.scale = scale

    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        """Compute dL/dx; the non-Tensor scale argument has no gradient."""
        (input_tensor,) = ctx.saved_tensors
        grad_input = (
            grad_output.unsqueeze(-1) * (2.0 * ctx.scale * input_tensor)
        )
        return grad_input, None

    torch.library.register_autograd(
        QUALIFIED_NAME,
        backward,
        setup_context=setup_context,
    )


_register_operator_once()
OP = _lookup_operator()
if OP is None:  # Defensive: registration should have created this overload.
    raise RuntimeError(f"failed to register {QUALIFIED_NAME}")


def rowwise_scaled_square(input_tensor: Tensor, scale: float = 1.0) -> Tensor:
    """Call the registered operator through torch.ops and the Dispatcher."""
    return OP(input_tensor, float(scale))


def fake_tensor_check() -> dict[str, Any]:
    """Run the operator on a FakeTensor and return observable metadata."""
    try:
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    except ImportError as error:  # The direct diagnostic import is version-sensitive.
        raise RuntimeError("FakeTensorMode is unavailable in this PyTorch build") from error

    with FakeTensorMode() as mode:
        fake_input = mode.from_tensor(torch.empty(4, 3, dtype=torch.float32))
        fake_output = rowwise_scaled_square(fake_input, 2.0)
        if not isinstance(fake_output, FakeTensor):
            raise AssertionError("registered fake kernel did not return a FakeTensor")
        return {
            "shape": tuple(fake_output.shape),
            "dtype": str(fake_output.dtype),
            "device": str(fake_output.device),
        }


def run_opcheck() -> dict[str, str] | None:
    """Run registration checks when torch.library.opcheck is available."""
    opcheck = getattr(torch.library, "opcheck", None)
    if opcheck is None:
        return None
    sample = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    return opcheck(OP, (sample, 1.25))


def run_checks() -> dict[str, Any]:
    """Exercise CPU, FakeTensor, Autograd, opcheck, and failure paths."""
    input_tensor = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    output = rowwise_scaled_square(input_tensor, 0.5)
    expected = torch.tensor([7.0, 0.625], dtype=torch.double)
    torch.testing.assert_close(output, expected)

    output.sum().backward()
    expected_gradient = input_tensor.detach()
    torch.testing.assert_close(input_tensor.grad, expected_gradient)

    gradcheck_input = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    gradcheck_passed = torch.autograd.gradcheck(
        lambda value: rowwise_scaled_square(value, 1.25),
        (gradcheck_input,),
    )

    fake = fake_tensor_check()
    opcheck_result = run_opcheck()

    failures: dict[str, str] = {}
    for name, bad_input in {
        "shape": torch.ones(3, dtype=torch.float32),
        "dtype": torch.ones(2, 3, dtype=torch.int64),
    }.items():
        try:
            rowwise_scaled_square(bad_input)
        except (TypeError, ValueError) as error:
            failures[name] = str(error)
        else:
            raise AssertionError(f"expected {name} contract violation to fail")

    return {
        "operator": QUALIFIED_NAME,
        "cpu_output": output.detach().tolist(),
        "gradient": input_tensor.grad.tolist(),
        "fake_tensor": fake,
        "gradcheck": gradcheck_passed,
        "opcheck": opcheck_result,
        "failures": failures,
    }


def main() -> None:
    result = run_checks()
    print("operator:", result["operator"])
    print("CPU output:", result["cpu_output"])
    print("Autograd gradient:", result["gradient"])
    print("FakeTensor metadata:", result["fake_tensor"])
    print("gradcheck:", result["gradcheck"])
    if result["opcheck"] is None:
        print("opcheck: unavailable in this PyTorch; registration checks skipped")
    else:
        print("opcheck:", result["opcheck"])
    print("rejected error paths:", sorted(result["failures"]))
    print("all custom-operator checks passed; no performance was measured")


if __name__ == "__main__":
    main()
