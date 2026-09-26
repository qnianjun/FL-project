"""Synthetic Sybil checks; no MNIST downloads or model training."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import TensorDataset

import run_sybil_backdoor as stage


class SybilTests(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(
            torch.zeros(100, 1, 28, 28), torch.arange(100) % 10
        )
        self.dataset.data, self.dataset.targets = self.dataset.tensors
        self.pair = (self.dataset, self.dataset)
        self.parts = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]

    def test_coalition_weight_matches_real_fedavg(self):
        for count in stage.SYBIL_COUNTS:
            mapping = stage.partition_mapping(count)
            updates = [
                ([np.array([1.0 if index == 0 else 0.0])], 12000)
                for index in mapping
            ]
            actual = aggregate(updates)[0][0]
            self.assertAlmostEqual(actual, count / (4 + count))
            self.assertEqual(
                stage.coalition_weight(count), count / (4 + count)
            )
        for count in (0, 4, -1, True, 1.5):
            with self.assertRaises(ValueError):
                stage.coalition_weight(count)

    def test_duplicate_indices_poisoning_and_paired_shuffle(self):
        for count in stage.SYBIL_COUNTS:
            clean, logical_parts, selected = stage.make_workers(
                self.pair, self.parts, 42, False, count
            )
            attack, repeated_parts, repeated = stage.make_workers(
                self.pair, self.parts, 42, True, count
            )
            self.assertEqual(logical_parts, repeated_parts)
            np.testing.assert_array_equal(selected, repeated)
            self.assertEqual(
                stage.pilot.identity(clean, logical_parts),
                stage.pilot.identity(attack, repeated_parts),
            )
            for workers, poisoned in ((clean, False), (attack, True)):
                stage.verify_workers(
                    workers, self.parts, selected, poisoned, count
                )
                stage.pilot.verify_transforms(
                    workers, self.pair, selected, poisoned
                )
                self.assertEqual(
                    len({id(w.model) for w in workers}), 4 + count
                )
                attackers = [w for w in workers if w.cid == 0]
                self.assertEqual(len(attackers), count)
                self.assertEqual(
                    {
                        index
                        for w in attackers
                        for index in (
                            w.trainset.dataset.indices
                            if poisoned
                            else w.trainset.indices
                        )
                    },
                    set(self.parts[0]),
                )
            for _ in range(2):
                clean_orders = [list(w.trainloader.sampler) for w in clean]
                attack_orders = [list(w.trainloader.sampler) for w in attack]
                self.assertEqual(clean_orders, attack_orders)
                for logical_id in range(5, 4 + count):
                    self.assertEqual(
                        attack_orders[0], attack_orders[logical_id]
                    )
        self.assertEqual(torch.count_nonzero(self.dataset.data).item(), 0)

    def test_single_identity_matches_original_pilot(self):
        for attack in (False, True):
            old, selected = stage.pilot.make_workers(
                self.pair, self.parts, 43, attack
            )
            new, parts, repeated = stage.make_workers(
                self.pair, self.parts, 43, attack, 1
            )
            self.assertEqual(
                stage.pilot.identity(old, self.parts),
                stage.pilot.identity(new, parts),
            )
            np.testing.assert_array_equal(selected, repeated)
            for left, right in zip(old, new):
                self.assertEqual(
                    list(left.trainloader.sampler),
                    list(right.trainloader.sampler),
                )
                self.assertEqual(type(left), type(right))

    def test_wrong_replica_data_or_shuffle_is_rejected(self):
        workers, _, selected = stage.make_workers(
            self.pair, self.parts, 42, True, 2
        )
        workers[-1].trainset.dataset.indices = self.parts[1]
        with self.assertRaises(ValueError):
            stage.verify_workers(workers, self.parts, selected, True, 2)
        workers, _, selected = stage.make_workers(
            self.pair, self.parts, 42, True, 2
        )
        workers[-1].trainloader.generator.manual_seed(999)
        with self.assertRaises(ValueError):
            stage.verify_workers(workers, self.parts, selected, True, 2)

    def test_summary_uses_seed_pairs_and_sample_sd(self):
        rows = []
        for count in stage.SYBIL_COUNTS:
            final = []
            for index, seed in enumerate((42, 43, 44)):
                for mode in ("clean", "backdoor"):
                    attack = mode == "backdoor"
                    final.append(
                        {
                            "seed": seed,
                            "condition": mode,
                            "round": 10,
                            "clean_loss": 0.5 + (0.1 * index if attack else 0),
                            "clean_accuracy_percent": 90
                            - (index + 1 if attack else 0),
                            "asr_percent": 1 + (2 * index if attack else 0),
                        }
                    )
            rows.extend(stage.paired_rows(count, list(reversed(final))))
        for row in stage.summary_rows(rows):
            self.assertEqual(row["paired_clean_accuracy_change_pp_mean"], -2)
            self.assertEqual(
                row["paired_clean_accuracy_change_pp_sample_sd"], 1
            )
            self.assertEqual(row["paired_asr_increase_pp_mean"], 2)
            self.assertEqual(row["paired_asr_increase_pp_sample_sd"], 2)
            self.assertEqual(row["attacker_coalition_weight_sample_sd"], 0)
        with self.assertRaises(ValueError):
            stage.summary_rows(rows[:-1])

    def test_saved_pipeline_with_mock_metrics_and_no_training(self):
        """Check serialization and completion with tiny fixtures."""
        count = 2
        expected = {}
        configs = {}
        for seed in (42, 43, 44):
            expected[seed] = (self.parts, None, None)
            for attack in (False, True):
                workers, _, selected = stage.make_workers(
                    self.pair, self.parts, seed, attack, count
                )
                configs[count, seed, attack] = stage.condition_config(
                    workers, self.parts, selected, seed, attack, count
                )

        def fake_metrics(workers, loader, folder, seed, mode):
            metric = {
                "clean_loss": 0.5,
                "clean_accuracy_percent": 90.0,
                "asr_percent": 60.0 if mode == "backdoor" else 1.0,
            }
            stage.sweep.write_csv(
                folder / "results.csv",
                [{"round": number, **metric} for number in range(1, 11)],
            )
            return {"seed": seed, "condition": mode, "round": 10, **metric}

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "sybil_2"
            hashes = {
                "loaded_arrays_sha256": stage.pilot.dataset_hashes(self.pair)
            }
            with patch.object(
                stage.pilot, "train_condition", side_effect=fake_metrics
            ), patch.object(
                stage.pilot.client, "train", side_effect=AssertionError
            ):
                final = stage.train_setting(
                    folder, count, self.pair, hashes, expected, configs
                )
            self.assertEqual(len(final), 6)
            with self.assertRaises(FileExistsError):
                stage.train_setting(
                    folder, count, self.pair, hashes, expected, configs
                )
            (folder / "seed_42/clean/results.csv").unlink()
            with self.assertRaises(ValueError):
                stage.validate_new_result(folder, count, expected, configs)

    def test_preview_does_not_train_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            with (
                patch.object(
                    stage.sys, "argv", ["stage", "--output", str(output)]
                ),
                patch.object(
                    stage.sweep, "current_data", return_value=(self.pair, {})
                ),
                patch.object(stage, "preflight", return_value=({}, {}, [])),
                patch.object(
                    stage, "train_setting", side_effect=AssertionError
                ),
                patch.object(
                    stage, "save_provenance", side_effect=AssertionError
                ),
            ):
                stage.main()
            self.assertFalse(output.exists())

    def test_existing_output_stops_before_any_data_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    stage.sys,
                    "argv",
                    ["stage", "--run", "--output", directory],
                ),
                patch.object(
                    stage.sweep, "current_data", side_effect=AssertionError
                ),
            ):
                with self.assertRaises(SystemExit) as stopped:
                    stage.main()
            self.assertEqual(stopped.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
