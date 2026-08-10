"""Tests for examples/pytorch/minigpt_project.py."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

import minigpt_project as project  # noqa: E402


class MiniGPTProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.vocabulary, self.dataset = project.build_project()
        project.set_determinism(self.config.seed)

    def make_training_objects(self):
        model = project.MiniGPT(self.config)
        optimizer = project._make_optimizer(model, self.config)
        policy = project.make_amp_policy(torch.device("cpu"))
        scaler = project.make_grad_scaler(policy)
        return model, optimizer, policy, scaler

    def test_vocabulary_is_deterministic_and_targets_shift_one_character(self) -> None:
        second = project.CharacterVocabulary.from_text(project.TEACHING_CORPUS)
        self.assertEqual(self.vocabulary.characters, second.characters)
        self.assertEqual(
            self.vocabulary.characters,
            tuple(sorted(set(project.TEACHING_CORPUS))),
        )
        inputs, targets = self.dataset[0]
        self.assertEqual(tuple(inputs.shape), (self.config.block_size,))
        self.assertTrue(torch.equal(inputs[1:], targets[:-1]))
        self.assertEqual(
            self.vocabulary.decode(inputs.tolist()),
            project.TEACHING_CORPUS[: self.config.block_size],
        )
        self.assertEqual(
            self.vocabulary.decode(targets.tolist()),
            project.TEACHING_CORPUS[1 : self.config.block_size + 1],
        )
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            self.vocabulary.decode([-1])
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            self.vocabulary.decode([len(self.vocabulary.characters)])

    def test_causal_mask_and_logits_shape(self) -> None:
        model = project.MiniGPT(self.config)
        mask = model.causal_mask(self.config.block_size)
        expected = torch.triu(
            torch.ones(
                self.config.block_size,
                self.config.block_size,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(mask.diagonal().any())
        inputs, _ = project._first_batch(self.dataset, self.config)
        logits = model(inputs)
        self.assertEqual(
            tuple(logits.shape),
            (self.config.batch_size, self.config.block_size, self.config.vocab_size),
        )
        with self.assertRaisesRegex(ValueError, "exceeds configured block_size"):
            model(torch.zeros(1, self.config.block_size + 1, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "profiler_repeat=1"):
            project.replace(self.config, profiler_repeat=2).validate()

    def test_cpu_step_has_finite_loss_and_updates_parameters(self) -> None:
        model, optimizer, policy, scaler = self.make_training_objects()
        self.assertFalse(policy.enabled)
        self.assertEqual(policy.autocast_dtype, "float32")
        batch = project._first_batch(self.dataset, self.config)
        before = project.flatten_parameters(model)
        observation = project.train_one_step(
            model,
            optimizer,
            scaler,
            batch,
            self.config,
            torch.device("cpu"),
            policy,
        )
        after = project.flatten_parameters(model)
        self.assertTrue(observation["finite_loss"])
        self.assertTrue(observation["parameter_updated"])
        self.assertTrue(torch.isfinite(torch.tensor(observation["loss"])))
        self.assertFalse(torch.equal(before, after))

    def test_checkpoint_restores_component_state_rng_and_can_resume(self) -> None:
        model, optimizer, policy, scaler = self.make_training_objects()
        batch = project._first_batch(self.dataset, self.config)
        project.train_one_step(
            model,
            optimizer,
            scaler,
            batch,
            self.config,
            torch.device("cpu"),
            policy,
        )
        with tempfile.TemporaryDirectory(prefix="minigpt-checkpoint-test-") as directory:
            path = Path(directory) / "checkpoint.pt"
            project.save_checkpoint(
                path,
                model,
                optimizer,
                scaler,
                self.config,
                step=1,
                vocabulary=self.vocabulary,
                device=torch.device("cpu"),
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(payload), project.required_checkpoint_fields())
            expected_python = random.random()
            expected_torch = torch.rand(3)

            resumed_model, resumed_optimizer, resumed_policy, resumed_scaler = (
                self.make_training_objects()
            )
            random.random()
            torch.rand(3)
            restored_step = project.load_checkpoint(
                path,
                resumed_model,
                resumed_optimizer,
                resumed_scaler,
                self.config,
                self.vocabulary,
                torch.device("cpu"),
            )
            self.assertEqual(restored_step, 1)
            self.assertTrue(project._state_dicts_equal(model, resumed_model))
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            resumed = project.train_one_step(
                resumed_model,
                resumed_optimizer,
                resumed_scaler,
                batch,
                self.config,
                torch.device("cpu"),
                resumed_policy,
            )
            self.assertTrue(resumed["parameter_updated"])

    def test_profiler_exports_parseable_trace_with_project_marker(self) -> None:
        model, optimizer, policy, scaler = self.make_training_objects()
        with tempfile.TemporaryDirectory(prefix="minigpt-profiler-test-") as directory:
            trace = Path(directory) / "trace.json"
            evidence = project.run_profiler(
                model,
                optimizer,
                scaler,
                self.dataset,
                self.config,
                torch.device("cpu"),
                policy,
                trace,
            )
            self.assertTrue(trace.exists())
            self.assertGreater(evidence["trace_events"], 0)
            self.assertEqual(evidence["activities"], ["CPU"])
            self.assertFalse(evidence["performance_measured"])
            payload = json.loads(trace.read_text(encoding="utf-8"))
            names = {event.get("name") for event in payload["traceEvents"]}
            self.assertIn("minigpt.train_step", names)

    def test_memory_report_has_logical_ledger_and_explicit_cpu_fallback(self) -> None:
        model, optimizer, policy, scaler = self.make_training_objects()
        project.train_one_step(
            model,
            optimizer,
            scaler,
            project._first_batch(self.dataset, self.config),
            self.config,
            torch.device("cpu"),
            policy,
        )
        report = project.memory_report(
            model,
            optimizer,
            self.config,
            torch.device("cpu"),
            policy,
        )
        project.validate_memory_report(report)
        self.assertGreater(report["logical_bytes"]["parameters"], 0)
        self.assertGreater(report["logical_bytes"]["gradients"], 0)
        self.assertGreater(report["logical_bytes"]["optimizer_state"], 0)
        self.assertGreater(report["logical_bytes"]["activations_estimate"], 0)
        self.assertTrue(report["activation_estimate"]["is_estimate"])
        self.assertFalse(report["cuda"]["measured"])
        self.assertIsNone(report["cuda"]["peak_allocated_bytes"])
        self.assertIn("GPU memory behavior was not measured", report["cuda"]["note"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA checkpoint path is optional")
class MiniGPTCudaCheckpointTest(unittest.TestCase):
    def test_cuda_checkpoint_restores_cpu_and_cuda_rng_state(self) -> None:
        config, vocabulary, dataset = project.build_project()
        project.set_determinism(config.seed)
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(config.seed)
        model = project.MiniGPT(config).to(device)
        optimizer = project._make_optimizer(model, config)
        policy = project.make_amp_policy(device)
        scaler = project.make_grad_scaler(policy)
        project.train_one_step(
            model,
            optimizer,
            scaler,
            project._first_batch(dataset, config),
            config,
            device,
            policy,
        )

        with tempfile.TemporaryDirectory(prefix="minigpt-cuda-checkpoint-test-") as directory:
            path = Path(directory) / "checkpoint.pt"
            project.save_checkpoint(
                path,
                model,
                optimizer,
                scaler,
                config,
                step=1,
                vocabulary=vocabulary,
                device=device,
            )
            expected_cuda_random = torch.rand(4, device=device)
            torch.rand(4, device=device)
            resumed_model = project.MiniGPT(config).to(device)
            resumed_optimizer = project._make_optimizer(resumed_model, config)
            resumed_scaler = project.make_grad_scaler(policy)
            restored_step = project.load_checkpoint(
                path,
                resumed_model,
                resumed_optimizer,
                resumed_scaler,
                config,
                vocabulary,
                device,
            )
            self.assertEqual(restored_step, 1)
            self.assertTrue(project._state_dicts_equal(model, resumed_model))
            self.assertTrue(
                torch.equal(torch.rand(4, device=device), expected_cuda_random)
            )
            floating_state_devices = {
                value.device.type
                for state in resumed_optimizer.state.values()
                for name, value in state.items()
                if name in {"exp_avg", "exp_avg_sq"} and isinstance(value, torch.Tensor)
            }
            self.assertEqual(floating_state_devices, {"cuda"})


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(),
    "two-process integration requires torch.distributed with Gloo",
)
class MiniGPTGlooIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory(
            prefix="minigpt-gloo-test-"
        )
        try:
            cls.evidence = project.run_ddp_smoke(cls.temporary_directory.name)
        except Exception as error:
            cls.temporary_directory.cleanup()
            raise RuntimeError(
                "Gloo is available, but the two-process MiniGPT integration failed; "
                "the environment may forbid process spawning, loopback communication, "
                "or temporary-file writes"
            ) from error

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_sampler_parameters_checkpoint_and_cleanup(self) -> None:
        rank0, rank1 = self.evidence["rank_indices"]
        self.assertTrue(self.evidence["sampler_set_epoch"])
        self.assertTrue(set(rank0).isdisjoint(rank1))
        self.assertEqual(
            sorted(rank0 + rank1),
            list(range(project.Config().ddp_dataset_size)),
        )
        self.assertEqual(self.evidence["parameter_max_diff"], 0.0)
        self.assertEqual(
            self.evidence["rank0_checkpoint_files"],
            ["minigpt_ddp_checkpoint.pt"],
        )
        self.assertEqual(self.evidence["destroyed_ranks"], [0, 1])
        self.assertEqual(
            set(self.evidence["checkpoint_complete_fields"]),
            project.required_checkpoint_fields(),
        )
        self.assertFalse(self.evidence["performance_measured"])


if __name__ == "__main__":
    unittest.main()
