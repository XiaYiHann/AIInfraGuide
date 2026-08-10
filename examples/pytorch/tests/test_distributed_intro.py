"""CPU/Gloo tests for examples/pytorch/distributed_intro.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch.distributed as dist

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import distributed_intro  # noqa: E402


class DistributedIntroTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed is unavailable")
        if not dist.is_gloo_available():
            raise unittest.SkipTest("Gloo is unavailable in this PyTorch build")

        cls.temporary_directory = tempfile.TemporaryDirectory(
            prefix="distributed-intro-test-"
        )
        try:
            cls.result = distributed_intro.run_distributed_smoke(
                cls.temporary_directory.name
            )
        except Exception as error:
            cls.temporary_directory.cleanup()
            raise RuntimeError(
                "Gloo is available, but the two-process smoke failed. The environment "
                "may forbid process spawning, loopback communication, or temporary-file writes."
            ) from error

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_distributed_sampler_shards_are_disjoint_and_cover_dataset(self) -> None:
        rank0, rank1 = self.result["rank_indices"]
        self.assertTrue(set(rank0).isdisjoint(rank1))
        self.assertEqual(
            sorted(rank0 + rank1),
            list(range(distributed_intro.DATASET_SIZE)),
        )
        self.assertTrue(self.result["shards_disjoint"])
        self.assertTrue(self.result["shards_cover_dataset"])

    def test_ddp_parameters_and_all_reduce_metric_match_across_ranks(self) -> None:
        self.assertEqual(self.result["world_size"], 2)
        self.assertEqual(self.result["backend"], "gloo")
        self.assertEqual(self.result["parameter_max_diff"], 0.0)
        self.assertEqual(self.result["metric_max_diff"], 0.0)
        self.assertGreaterEqual(self.result["global_mean_loss"], 0.0)
        self.assertFalse(self.result["performance_measured"])

    def test_only_rank_zero_checkpoints_and_every_rank_cleans_up(self) -> None:
        self.assertEqual(self.result["checkpoint_files"], ["rank0_checkpoint.pt"])
        self.assertEqual(self.result["checkpoint_saved_by_rank"], 0)
        self.assertEqual(self.result["destroyed_ranks"], [0, 1])

    def test_output_directory_can_be_reused_without_touching_unrelated_files(self) -> None:
        directory = Path(self.temporary_directory.name)
        unrelated = directory / "other_checkpoint.pt"
        unrelated.write_text("owned by another experiment", encoding="utf-8")
        repeated = distributed_intro.run_distributed_smoke(directory)
        self.assertEqual(repeated["checkpoint_files"], ["rank0_checkpoint.pt"])
        self.assertEqual(
            unrelated.read_text(encoding="utf-8"),
            "owned by another experiment",
        )

    def test_torchrun_environment_parser_keeps_global_and_local_rank_distinct(self) -> None:
        parsed = distributed_intro.read_torchrun_environment(
            {
                "RANK": "3",
                "LOCAL_RANK": "1",
                "WORLD_SIZE": "4",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
        )
        self.assertEqual(parsed["rank"], 3)
        self.assertEqual(parsed["local_rank"], 1)
        self.assertEqual(parsed["world_size"], 4)

        with self.assertRaisesRegex(RuntimeError, "missing: MASTER_PORT"):
            distributed_intro.read_torchrun_environment(
                {
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": "2",
                    "MASTER_ADDR": "127.0.0.1",
                }
            )


if __name__ == "__main__":
    unittest.main()
