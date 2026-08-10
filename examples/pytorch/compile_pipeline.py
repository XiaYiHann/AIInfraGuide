"""CPU probes for the PyTorch execution/compilation pipeline in lesson 4.10.

Run from the repository root:
    python3 examples/pytorch/compile_pipeline.py

The default backend is ``eager`` on purpose: it exercises TorchDynamo capture,
guards, graph-break handling, and ``torch.compile``'s callable contract without
paying TorchInductor code-generation cost. This is a correctness/diagnostic
example, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import torch
from torch import Tensor, nn


DEFAULT_BACKEND = "eager"
SEED = 20260810


def _require_compile_apis() -> Any:
    """Return torch._dynamo or fail with an actionable version message."""
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "torch.compile is unavailable; install a PyTorch 2.x build that "
            "provides the torch.compiler stack"
        )
    try:
        import torch._dynamo as dynamo
    except ImportError as error:
        raise RuntimeError(
            "torch._dynamo is unavailable in this PyTorch build; the 4.10 "
            "Dynamo diagnostics cannot run"
        ) from error

    missing = [name for name in ("explain", "graph_break") if not hasattr(dynamo, name)]
    if missing:
        raise RuntimeError(
            "this PyTorch build is missing required Dynamo diagnostics: "
            + ", ".join(missing)
        )
    return dynamo


def tiny_pipeline(input_tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """A small differentiable Tensor program with no data-dependent Python."""
    hidden = input_tensor @ weight + bias
    activated = torch.tanh(hidden)
    return activated.square().mean(dim=-1)


class ExportableTinyPipeline(nn.Module):
    """Module wrapper because torch.export.export consumes an nn.Module."""

    def forward(self, input_tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        return tiny_pipeline(input_tensor, weight, bias)


def intentional_graph_break(input_tensor: Tensor) -> Tensor:
    """Split an otherwise traceable function at an explicit diagnostic break."""
    prefix = torch.sin(input_tensor)
    torch._dynamo.graph_break()
    return torch.cos(prefix)


def make_inputs(batch_size: int = 3) -> tuple[Tensor, Tensor, Tensor]:
    """Create fixed CPU inputs; reject invalid shapes before compilation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(SEED + batch_size)
    return (
        torch.randn(batch_size, 4, generator=generator),
        torch.randn(4, 5, generator=generator),
        torch.randn(5, generator=generator),
    )


def _clone_for_grad(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


def _compile(
    function: Callable[..., Tensor],
    *,
    backend: str = DEFAULT_BACKEND,
    fullgraph: bool = False,
    dynamic: bool | None = None,
) -> Callable[..., Tensor]:
    """Compile a callable and translate API/backend failures into clear errors."""
    _require_compile_apis()
    try:
        return torch.compile(
            function,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
    except (TypeError, RuntimeError) as error:
        raise RuntimeError(
            "torch.compile setup failed for "
            f"backend={backend!r}, fullgraph={fullgraph}, dynamic={dynamic}: {error}"
        ) from error


def compare_eager_and_compile(backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Verify forward values and all input gradients against eager execution."""
    dynamo = _require_compile_apis()
    dynamo.reset()
    base_inputs = make_inputs()
    eager_inputs = _clone_for_grad(base_inputs)
    compiled_inputs = _clone_for_grad(base_inputs)

    eager_output = tiny_pipeline(*eager_inputs)
    eager_output.sum().backward()

    compiled_function = _compile(tiny_pipeline, backend=backend)
    compiled_output = compiled_function(*compiled_inputs)
    compiled_output.sum().backward()

    torch.testing.assert_close(compiled_output, eager_output, rtol=1e-5, atol=1e-6)
    gradient_errors: list[float] = []
    for eager_tensor, compiled_tensor in zip(eager_inputs, compiled_inputs):
        if eager_tensor.grad is None or compiled_tensor.grad is None:
            raise AssertionError("eager and compiled paths must populate every input gradient")
        torch.testing.assert_close(
            compiled_tensor.grad,
            eager_tensor.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        gradient_errors.append(
            float((compiled_tensor.grad - eager_tensor.grad).abs().max().item())
        )

    return {
        "backend": backend,
        "output_shape": list(compiled_output.shape),
        "max_output_abs_error": float((compiled_output - eager_output).abs().max().item()),
        "max_gradient_abs_errors": gradient_errors,
    }


def _run_explain(function: Callable[..., Tensor], inputs: tuple[Tensor, ...]) -> Any:
    """Support the current curried explain API and report older-API fallback."""
    dynamo = _require_compile_apis()
    dynamo.reset()
    try:
        return dynamo.explain(function)(*inputs)
    except TypeError as current_error:
        try:
            return dynamo.explain(function, *inputs)
        except TypeError as legacy_error:
            raise RuntimeError(
                "torch._dynamo.explain is present but neither the current "
                "explain(fn)(*args) nor legacy explain(fn, *args) API worked"
            ) from legacy_error
        except Exception:
            raise
    except Exception:
        raise


def explain_summary(
    function: Callable[..., Tensor] = tiny_pipeline,
    inputs: tuple[Tensor, ...] | None = None,
) -> dict[str, Any]:
    """Return a version-tolerant graph/guard/break summary from Dynamo explain."""
    if inputs is None:
        inputs = make_inputs()
    explanation = _run_explain(function, inputs)

    guards = getattr(explanation, "out_guards", None)
    if guards is None:
        guards = getattr(explanation, "guards", ())
    break_reasons = getattr(explanation, "break_reasons", ()) or ()

    def compact(value: Any, limit: int = 240) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    return {
        "graph_count": int(getattr(explanation, "graph_count", 0)),
        "graph_break_count": int(getattr(explanation, "graph_break_count", 0)),
        "operation_count": int(getattr(explanation, "op_count", 0)),
        "guard_count": len(guards),
        "guard_samples": [compact(guard) for guard in list(guards)[:3]],
        "break_reasons": [compact(reason) for reason in break_reasons],
    }


def fullgraph_failure_probe() -> dict[str, str]:
    """Prove that fullgraph=True rejects the explicit graph break."""
    dynamo = _require_compile_apis()
    dynamo.reset()
    compiled_function = _compile(
        intentional_graph_break,
        backend=DEFAULT_BACKEND,
        fullgraph=True,
    )
    try:
        compiled_function(torch.randn(3, 4))
    except Exception as error:  # Exception class is internal and version-sensitive.
        return {
            "status": "rejected_as_expected",
            "error_type": type(error).__name__,
            "message": " ".join(str(error).split())[:500],
        }
    raise AssertionError("fullgraph=True unexpectedly accepted an explicit graph break")


def dynamic_shape_probe(backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Exercise the public dynamic=True entry with two batch dimensions."""
    dynamo = _require_compile_apis()
    dynamo.reset()
    compiled_function = _compile(tiny_pipeline, backend=backend, dynamic=True)
    observed_shapes: list[list[int]] = []

    for batch_size in (2, 5):
        inputs = make_inputs(batch_size)
        eager_output = tiny_pipeline(*inputs)
        compiled_output = compiled_function(*inputs)
        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-5, atol=1e-6)
        observed_shapes.append(list(compiled_output.shape))

    return {
        "backend": backend,
        "configuration": "torch.compile(..., dynamic=True)",
        "input_batch_sizes": [2, 5],
        "output_shapes": observed_shapes,
        "claim": "correctness_only; recompilation count and speed were not asserted",
    }


def export_summary() -> dict[str, Any]:
    """Capture an ExportedProgram when torch.export is available, else degrade clearly."""
    export_namespace = getattr(torch, "export", None)
    export_function = getattr(export_namespace, "export", None)
    if export_function is None:
        return {
            "available": False,
            "reason": "torch.export.export is unavailable in this PyTorch build",
        }

    inputs = make_inputs()
    try:
        exported_program = export_function(ExportableTinyPipeline(), inputs, strict=True)
    except (TypeError, RuntimeError) as error:
        raise RuntimeError(
            "torch.export.export exists but failed to capture the tiny Tensor program: "
            f"{error}"
        ) from error

    graph = exported_program.graph_module.graph
    return {
        "available": True,
        "node_count": sum(1 for _ in graph.nodes),
        "graph_signature": str(exported_program.graph_signature),
        "range_constraint_count": len(exported_program.range_constraints),
    }


def run_checks(backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Run all CPU correctness and diagnostic probes; never measure speed."""
    eager_compile = compare_eager_and_compile(backend)
    explain = explain_summary()
    if explain["graph_count"] < 1 or explain["guard_count"] < 1:
        raise AssertionError("Dynamo explain did not report a captured graph and guards")
    if explain["graph_break_count"] != 0:
        raise AssertionError("the normal tiny pipeline should not contain graph breaks")

    break_explain = explain_summary(
        intentional_graph_break,
        (torch.randn(3, 4),),
    )
    if break_explain["graph_break_count"] < 1:
        raise AssertionError("the intentional break was not reported by Dynamo explain")

    return {
        "torch_version": torch.__version__,
        "device": "cpu",
        "scope": "correctness_and_diagnostics_only_no_performance_claim",
        "eager_compile_equivalence": eager_compile,
        "dynamo_explain": explain,
        "intentional_graph_break": break_explain,
        "fullgraph_failure": fullgraph_failure_probe(),
        "dynamic_shapes": dynamic_shape_probe(backend),
        "export": export_summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CPU correctness probes for the torch.compile pipeline."
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="torch.compile backend (default: eager, chosen for a light smoke run)",
    )
    arguments = parser.parse_args()
    result = run_checks(arguments.backend)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("all compile-pipeline checks passed; no performance was measured")


if __name__ == "__main__":
    main()
