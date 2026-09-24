"""Quick scalar and integration checks; no FL training or downloads."""

import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

import run_update_similarity as stage


class SimilarityTests(unittest.TestCase):
    def test_full_model_norm_cosine_zero_and_identical(self):
        initial = [np.array([10.0, 20.0]), np.array([[30.0]])]
        deltas = [(3, 4, 12), (3, 4, 12), (-3, -4, -12), (0, 0, 0)]
        submissions = [
            ([initial[0] + delta[:2], initial[1] + delta[2]], 12000)
            for delta in deltas
        ]
        before = [[value.copy() for value in p] for p, _ in submissions]
        norms, pairs = stage.update_statistics(initial, submissions)
        self.assertEqual(norms, [13, 13, 13, 0])
        values = {(i, j): value for i, j, value in pairs}
        self.assertAlmostEqual(values[0, 1], 1)
        self.assertAlmostEqual(values[0, 2], -1)
        self.assertIsNone(values[0, 3])
        self.assertIsNone(values[2, 3])
        for saved, (parameters, _) in zip(before, submissions):
            for left, right in zip(saved, parameters):
                np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(initial[0], [10, 20])

    def test_symmetry_and_orthogonal_vectors(self):
        initial = [np.zeros(2, dtype=np.float32)]
        submissions = [
            ([np.array(value)], 1)
            for value in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0])
        ]
        _, forward = stage.update_statistics(initial, submissions)
        _, reverse = stage.update_statistics(initial, submissions[::-1])
        reverse = {(2 - j, 2 - i): value for i, j, value in reverse}
        for i, j, value in forward:
            self.assertAlmostEqual(value, reverse[i, j])
        self.assertEqual(forward[0][2], 0)
        self.assertAlmostEqual(forward[1][2], 1 / np.sqrt(2))

    def test_pair_labels_and_counts(self):
        labels = [
            stage.pair_type(i, j) for i in range(7) for j in range(i + 1, 7)
        ]
        self.assertEqual(labels.count("honest_honest"), 6)
        self.assertEqual(labels.count("sybil_honest"), 12)
        self.assertEqual(labels.count("sybil_sybil"), 3)

    def test_logging_leaves_real_loop_outputs_and_rng_unchanged(self):
        """Real pilot loop and FedAvg, deterministic fake local training."""
        dataset = TensorDataset(
            torch.zeros(10, 1, 28, 28), torch.arange(10) % 10
        )
        dataset.data, dataset.targets = dataset.tensors
        parts = [list(range(i * 2, (i + 1) * 2)) for i in range(5)]

        def fake_train(model, loader):
            order = list(loader.sampler)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(
                        sum((i + 1) * n for i, n in enumerate(order)) * 0.0001
                    )

        for attack in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                captures = []
                for logging in (False, True):
                    folder = Path(temporary) / str(logging)
                    folder.mkdir()
                    workers, _, _ = stage.sybil.make_workers(
                        (dataset, dataset), parts, 42, attack, 3
                    )
                    with (
                        patch.object(
                            stage.pilot,
                            "SETTINGS",
                            replace(stage.pilot.SETTINGS, rounds=2),
                        ),
                        patch.object(stage.pilot.client, "train", fake_train),
                        patch.object(
                            stage.pilot.client, "test", return_value=(0.5, 0.8)
                        ),
                    ):
                        runner = (
                            stage.train_with_logging
                            if logging
                            else stage.pilot.train_condition
                        )
                        runner(workers, None, folder, 42, "fixture")
                    captures.append(
                        (
                            [w.get_parameters({}) for w in workers],
                            [
                                w.trainloader.generator.get_state()
                                for w in workers
                            ],
                            torch.get_rng_state(),
                            (folder / "results.csv").read_bytes(),
                        )
                    )
                for left, right in zip(captures[0][0], captures[1][0]):
                    for a, b in zip(left, right):
                        np.testing.assert_array_equal(a, b)
                for a, b in zip(captures[0][1], captures[1][1]):
                    self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(captures[0][2], captures[1][2]))
                self.assertEqual(captures[0][3], captures[1][3])
                with (folder / "pairwise_similarity.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 42)
                self.assertEqual(
                    set(rows[0]),
                    {
                        "round",
                        "client_i",
                        "client_j",
                        "pair_type",
                        "cosine_similarity",
                    },
                )
                with self.assertRaises(FileExistsError):
                    stage.train_with_logging(workers, None, folder, 42, "x")

    def test_preview_has_no_training_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            with (
                patch.object(
                    stage.sys, "argv", ["stage", "--output", str(output)]
                ),
                patch.object(
                    stage.sweep, "current_data", return_value=(None, None)
                ),
                patch.object(stage, "preflight", return_value=({}, {})),
                patch.object(stage, "run", side_effect=AssertionError),
            ):
                stage.main()
            self.assertFalse(output.exists())

    def test_existing_output_stops_before_loading_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    stage.sys,
                    "argv",
                    ["stage", "--run", "--output", temporary],
                ),
                patch.object(
                    stage.sweep, "current_data", side_effect=AssertionError
                ),
            ):
                with self.assertRaises(SystemExit):
                    stage.main()


if __name__ == "__main__":
    unittest.main()
