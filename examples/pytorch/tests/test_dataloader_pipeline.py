"""CPU tests for examples/pytorch/dataloader_pipeline.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import dataloader_pipeline  # noqa: E402


class DataLoaderPipelineTest(unittest.TestCase):
    def test_map_dataset_and_collate_pad_variable_lengths(self) -> None:
        dataset = dataloader_pipeline.MapSequenceDataset(
            [[1, 2, 3], [4], [5, 6]]
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[2]["tokens"].tolist(), [5, 6])

        batch = dataloader_pipeline.pad_collate([dataset[0], dataset[1]])
        self.assertEqual(batch.ids.tolist(), [0, 1])
        self.assertEqual(batch.tokens.tolist(), [[1, 2, 3], [4, 0, 0]])
        self.assertEqual(batch.lengths.tolist(), [3, 1])
        self.assertEqual(batch.targets.tolist(), [0, 1])

    def test_sampler_batch_sampler_and_drop_last_control_indices(self) -> None:
        self.assertEqual(
            dataloader_pipeline.sampler_batches(drop_last=False),
            [[4, 2], [0, 3], [1]],
        )
        self.assertEqual(
            dataloader_pipeline.sampler_batches(drop_last=True),
            [[4, 2], [0, 3]],
        )

    def test_iterable_dataset_shards_worker_replicas_without_duplicates(self) -> None:
        loader = DataLoader(
            dataloader_pipeline.ShardedRangeDataset(0, 17),
            batch_size=None,
            num_workers=2,
        )
        values = list(loader)
        self.assertEqual(sorted(values), list(range(17)))
        self.assertEqual(len(values), len(set(values)))

    def test_worker_seed_is_unique_per_worker_and_reproducible(self) -> None:
        first = dataloader_pipeline.collect_seed_records(num_workers=2)
        second = dataloader_pipeline.collect_seed_records(num_workers=2)
        self.assertEqual(first, second)
        self.assertEqual([record[0] for record in first], list(range(8)))
        self.assertEqual({record[1] for record in first}, {0, 1})
        self.assertEqual(len({record[2] for record in first}), 2)

    def test_invalid_sampler_configuration_and_collate_fail_loudly(self) -> None:
        message = dataloader_pipeline.invalid_sampler_configuration()
        self.assertIn("mutually exclusive", message)

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            dataloader_pipeline.pad_collate([])

        malformed = {"id": 0, "tokens": torch.empty(0, dtype=torch.long), "target": 0}
        with self.assertRaisesRegex(ValueError, "non-empty 1-D"):
            dataloader_pipeline.pad_collate([malformed])

    def test_no_cuda_path_degrades_explicitly_without_performance_claim(self) -> None:
        dataset = dataloader_pipeline.MapSequenceDataset([[1, 2], [3]])
        batch = dataloader_pipeline.pad_collate([dataset[0], dataset[1]])
        moved, message = dataloader_pipeline.transfer_batch(
            batch,
            request_cuda=True,
            cuda_available=False,
        )
        self.assertEqual(moved.tokens.device.type, "cpu")
        self.assertIn("CUDA unavailable", message)
        self.assertIn("performance were not measured", message)

    def test_loader_comparison_keeps_work_fixed_without_speed_threshold(self) -> None:
        observations = dataloader_pipeline.compare_loader_configs(pin_memory=False)
        self.assertEqual(len(observations), 3)
        self.assertEqual(
            [(item["num_workers"], item["prefetch_factor"], item["persistent_workers"]) for item in observations],
            [(0, None, False), (2, 1, False), (2, 2, True)],
        )
        for item in observations:
            self.assertEqual(item["batches"], 4)
            self.assertEqual(item["samples"], 16)
            self.assertGreater(item["seconds"], 0)
            self.assertGreater(item["samples_per_second"], 0)

    def test_loader_configuration_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_workers"):
            dataloader_pipeline.make_demo_loader(-1, pin_memory=False)
        with self.assertRaisesRegex(ValueError, "prefetch_factor"):
            dataloader_pipeline.make_demo_loader(2, pin_memory=False, prefetch_factor=0)

    def test_distributed_sampler_entry_partitions_a_divisible_dataset(self) -> None:
        rank0, rank1 = dataloader_pipeline.distributed_sampler_entry()
        self.assertEqual(sorted(rank0 + rank1), list(range(8)))
        self.assertTrue(set(rank0).isdisjoint(rank1))


if __name__ == "__main__":
    unittest.main()
