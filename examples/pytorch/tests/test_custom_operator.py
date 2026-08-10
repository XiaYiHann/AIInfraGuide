"""CPU tests for examples/pytorch/custom_operator.py."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import custom_operator  # noqa: E402


class CustomOperatorTest(unittest.TestCase):
    def test_operator_is_registered_once_with_cpu_dispatch(self) -> None:
        self.assertIsNotNone(custom_operator.OP)
        self.assertEqual(
            custom_operator.OP.name(),
            custom_operator.QUALIFIED_NAME,
        )

        # Reloading simulates duplicate test discovery/import paths. The guard
        # must reuse the existing operator rather than register its name again.
        reloaded = importlib.reload(custom_operator)
        self.assertEqual(reloaded.OP.name(), reloaded.QUALIFIED_NAME)
        result = reloaded.rowwise_scaled_square(torch.ones(2, 3), 1.0)
        torch.testing.assert_close(result, torch.full((2,), 3.0))

    def test_cpu_implementation_returns_expected_fresh_tensor(self) -> None:
        input_tensor = torch.tensor(
            [[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]],
            dtype=torch.float64,
        )
        output = custom_operator.rowwise_scaled_square(input_tensor, 0.5)
        expected = torch.tensor([7.0, 0.625], dtype=torch.float64)
        torch.testing.assert_close(output, expected)
        self.assertEqual(output.shape, (2,))
        self.assertEqual(output.dtype, input_tensor.dtype)
        self.assertEqual(output.device.type, "cpu")
        self.assertNotEqual(output.untyped_storage().data_ptr(), input_tensor.untyped_storage().data_ptr())

    def test_fake_tensor_kernel_propagates_metadata(self) -> None:
        try:
            metadata = custom_operator.fake_tensor_check()
        except RuntimeError as error:
            self.skipTest(str(error))
        self.assertEqual(metadata["shape"], (4,))
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["device"], "cpu")

    def test_registered_autograd_formula_and_gradcheck(self) -> None:
        input_tensor = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]],
            dtype=torch.double,
            requires_grad=True,
        )
        scale = 1.5
        custom_operator.rowwise_scaled_square(input_tensor, scale).sum().backward()
        torch.testing.assert_close(
            input_tensor.grad,
            2.0 * scale * input_tensor.detach(),
        )

        check_input = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: custom_operator.rowwise_scaled_square(value, scale),
                (check_input,),
            )
        )

    @unittest.skipUnless(
        hasattr(torch.library, "opcheck"),
        "torch.library.opcheck unavailable in this PyTorch",
    )
    def test_opcheck_validates_registration_contract(self) -> None:
        result = custom_operator.run_opcheck()
        self.assertIsNotNone(result)
        self.assertEqual(set(result.values()), {"SUCCESS"})
        self.assertIn("test_schema", result)
        self.assertIn("test_autograd_registration", result)
        self.assertIn("test_faketensor", result)

    def test_bad_shape_and_dtype_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 2-D"):
            custom_operator.rowwise_scaled_square(torch.ones(3))
        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            custom_operator.rowwise_scaled_square(
                torch.ones(2, 3, dtype=torch.int64)
            )


if __name__ == "__main__":
    unittest.main()
