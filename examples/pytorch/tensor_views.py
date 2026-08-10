"""CPU experiments for PyTorch tensor storage and view semantics."""

from __future__ import annotations

import itertools
from typing import Any

import torch


def shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two live CPU tensors use the same underlying storage."""
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def layout(tensor: torch.Tensor) -> dict[str, Any]:
    """Collect the metadata needed to explain a strided tensor."""
    return {
        "shape": tuple(tensor.shape),
        "stride": tensor.stride(),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
    }


def storage_indices(tensor: torch.Tensor) -> list[int]:
    """Map every logical index to its element offset in storage."""
    indices: list[int] = []
    for logical_index in itertools.product(*(range(size) for size in tensor.shape)):
        offset = tensor.storage_offset()
        offset += sum(index * stride for index, stride in zip(logical_index, tensor.stride()))
        indices.append(offset)
    return indices


def make_examples() -> dict[str, torch.Tensor]:
    """Build deterministic tensors used by the article and unit tests."""
    base = torch.arange(12, dtype=torch.int64, device="cpu").reshape(3, 4)
    viewed = base.view(2, 6)
    transposed = base.transpose(0, 1)
    permuted = base.reshape(1, 3, 4).permute(0, 2, 1)
    reshaped = transposed.reshape(-1)
    contiguous = transposed.contiguous()

    broadcast_source = torch.arange(3, dtype=torch.int64, device="cpu").view(3, 1)
    broadcast = broadcast_source.expand(3, 4)

    overlap_source = torch.arange(4, dtype=torch.int64, device="cpu")
    overlap = torch.as_strided(overlap_source, size=(3, 2), stride=(1, 1))

    return {
        "base": base,
        "viewed": viewed,
        "transposed": transposed,
        "permuted": permuted,
        "reshaped": reshaped,
        "contiguous": contiguous,
        "broadcast_source": broadcast_source,
        "broadcast": broadcast,
        "overlap_source": overlap_source,
        "overlap": overlap,
    }


def run_checks() -> None:
    """Assert the storage and layout invariants demonstrated in this lesson."""
    tensors = make_examples()
    base = tensors["base"]
    transposed = tensors["transposed"]

    assert layout(base) == {
        "shape": (3, 4),
        "stride": (4, 1),
        "storage_offset": 0,
        "is_contiguous": True,
    }
    assert shares_storage(base, tensors["viewed"])
    assert layout(transposed)["stride"] == (1, 4)
    assert not transposed.is_contiguous()
    assert shares_storage(base, transposed)
    assert shares_storage(base, tensors["permuted"])

    try:
        transposed.view(-1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("view() should reject this incompatible stride layout")

    assert tensors["reshaped"].is_contiguous()
    assert not shares_storage(base, tensors["reshaped"])
    assert tensors["contiguous"].is_contiguous()
    assert not shares_storage(base, tensors["contiguous"])
    assert base.contiguous() is base

    broadcast = tensors["broadcast"]
    assert broadcast.stride() == (1, 0)
    assert shares_storage(tensors["broadcast_source"], broadcast)
    assert torch.equal(broadcast[:, 0], broadcast[:, 3])

    overlap_offsets = storage_indices(tensors["overlap"])
    assert overlap_offsets == [0, 1, 1, 2, 2, 3]
    assert len(set(overlap_offsets)) < len(overlap_offsets)


def main() -> None:
    tensors = make_examples()
    run_checks()

    for name in ("base", "viewed", "transposed", "permuted", "reshaped", "contiguous"):
        print(f"{name:12s} {layout(tensors[name])}")

    print("view shares base storage:", shares_storage(tensors["base"], tensors["viewed"]))
    print(
        "transpose shares base storage:",
        shares_storage(tensors["base"], tensors["transposed"]),
    )
    print(
        "reshape(transpose) shares base storage:",
        shares_storage(tensors["base"], tensors["reshaped"]),
    )
    print("broadcast stride:", tensors["broadcast"].stride())
    print("overlap storage indices:", storage_indices(tensors["overlap"]))
    print("all tensor-view checks passed")


if __name__ == "__main__":
    main()
