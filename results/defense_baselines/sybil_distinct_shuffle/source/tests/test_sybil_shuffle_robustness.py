"""Different-shuffle invariants; synthetic data, no local training."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

import run_sybil_shuffle_robustness as runner
from similarity_group_aggregation import aggregate_grouped


class SyntheticMNIST(Dataset):
    """Indexable 60,000-sample fixture without allocating MNIST images."""

    def __len__(self):
        return 60000

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), index % 10


class ShuffleTests(unittest.TestCase):
    def setUp(self):
        dataset = SyntheticMNIST()
        self.pair = (dataset, dataset)
        self.parts = [
            list(range(i * 12000, (i + 1) * 12000)) for i in range(5)
        ]

    def test_distinct_deterministic_matched_streams_and_unchanged_data(self):
        for seed in (42, 43, 44):
            identities = []
            orders = []
            for attack in (False, True):
                old, old_parts, old_selected = runner.baseline.make_workers(
                    runner.stage.BENCHMARK, self.pair, self.parts, seed, attack
                )
                original = runner.run_config(
                    old, old_parts, old_selected, seed, attack
                )
                workers, parts, selected = runner.make_workers(
                    self.pair, self.parts, seed, attack
                )
                config = runner.verify_workers(
                    workers, parts, selected, seed, attack, original
                )
                self.assertEqual(config["unique_attacker_sample_count"], 12000)
                np.testing.assert_array_equal(old_selected, selected)
                self.assertEqual(old_parts, parts)
                self.assertEqual(len(selected), 2400)
                identities.append(runner.pilot.identity(workers, parts))
                for i in range(5):
                    self.assertTrue(
                        torch.equal(
                            old[i].trainloader.generator.get_state(),
                            workers[i].trainloader.generator.get_state(),
                        )
                    )
                    self.assertEqual(
                        old[i].trainloader.batch_size,
                        workers[i].trainloader.batch_size,
                    )
                    self.assertEqual(
                        type(old[i].trainset), type(workers[i].trainset)
                    )
                    for a, b in zip(
                        old[i].get_parameters({}),
                        workers[i].get_parameters({}),
                    ):
                        np.testing.assert_array_equal(a, b)
                current = [list(w.trainloader.sampler) for w in workers]
                self.assertNotEqual(current[0], current[5])
                self.assertNotEqual(current[0], current[6])
                self.assertNotEqual(current[5], current[6])
                repeated, _, _ = runner.make_workers(
                    self.pair, self.parts, seed, attack
                )
                self.assertEqual(
                    current, [list(w.trainloader.sampler) for w in repeated]
                )
                orders.append(current)
            self.assertEqual(identities[0], identities[1])
            self.assertEqual(orders[0], orders[1])

    def test_aggregation_has_no_label_input(self):
        initial = [np.zeros(2, dtype=np.float32)]
        submissions = [
            ([np.array([i, 1], dtype=np.float32)], 12000) for i in range(7)
        ]
        before = [[v.copy() for v in p] for p, _ in submissions]
        output, _, groups, _ = aggregate_grouped(initial, submissions)
        runner.stage.analyze_groups(groups, {0, 5, 6})
        runner.stage.analyze_groups(groups, {1, 2, 3})
        repeated, _, _, _ = aggregate_grouped(initial, submissions)
        np.testing.assert_array_equal(output[0], repeated[0])
        for saved, (parameters, _) in zip(before, submissions):
            np.testing.assert_array_equal(saved[0], parameters[0])
        with self.assertRaises(TypeError):
            aggregate_grouped(initial, submissions, labels={0, 5, 6})

    def test_pair_labels_are_post_processing_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            runner.sweep.write_csv(
                folder / "pairwise_similarity.csv",
                [
                    {
                        "round": 1,
                        "client_i": 0,
                        "client_j": 5,
                        "cosine_similarity": 0.4,
                    },
                    {
                        "round": 1,
                        "client_i": 1,
                        "client_j": 2,
                        "cosine_similarity": 0.5,
                    },
                    {
                        "round": 1,
                        "client_i": 0,
                        "client_j": 1,
                        "cosine_similarity": 0.6,
                    },
                ],
            )
            with patch.object(
                runner.stage, "train_condition", return_value=({"asr": 1}, {})
            ):
                metric, _ = runner.train_condition(None, True, folder, None)
            self.assertEqual(metric, {"asr": 1})
            labels = [
                r["pair_type"]
                for r in runner.stage.read_rows(
                    folder / "pairwise_similarity_labeled.csv"
                )
            ]
            self.assertEqual(
                labels, ["sybil_sybil", "honest_honest", "sybil_honest"]
            )

    def test_preview_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with (
                patch.object(
                    runner.sys, "argv", ["runner", "--output", str(output)]
                ),
                patch.object(
                    runner.sweep, "current_data", return_value=(None, None)
                ),
                patch.object(
                    runner, "preflight", return_value=({}, {}, {}, {}, {})
                ),
                patch.object(runner, "execute", side_effect=AssertionError),
            ):
                runner.main()
            self.assertFalse(output.exists())
            with (
                patch.object(
                    runner.sys,
                    "argv",
                    ["runner", "--run", "--output", directory],
                ),
                patch.object(
                    runner.sweep, "current_data", side_effect=AssertionError
                ),
            ):
                with self.assertRaises(SystemExit):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
