"""Defense math and orchestration checks; no training or downloads."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import TensorDataset

from defense_aggregation import aggregate_updates
import run_defense_pilot as pilot


class AggregationTests(unittest.TestCase):
    def test_clipping_full_model_norm_and_direction(self):
        global_parameters = [
            np.array([10, -5], dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        ]
        magnitudes = [1, 2, 3, 4, 1000]
        submissions = [
            (
                [
                    global_parameters[0]
                    + np.array([3, 4], dtype=np.float32) * value,
                    np.array([[12 * value]], dtype=np.float32),
                ],
                12000,
            )
            for value in magnitudes
        ]
        before = [[value.copy() for value in p] for p, _ in submissions]
        result, logs = aggregate_updates(
            global_parameters, submissions, "clip_fedavg"
        )
        # Norm spans both tensors: sqrt(3² + 4² + 12²) = 13.
        self.assertEqual(logs[-1]["threshold"], 39)
        self.assertAlmostEqual(logs[-1]["clipped_update_l2"], 39)
        self.assertAlmostEqual(
            logs[-1]["reconstructed_update_l2"], 39, places=5
        )
        self.assertAlmostEqual(logs[-1]["clip_factor"], 3 / 1000)
        for log in logs:
            self.assertLessEqual(log["clipped_update_l2"], 39 + 1e-10)
        np.testing.assert_allclose(
            result[0] - global_parameters[0], [7.2, 9.6], atol=1e-6
        )
        np.testing.assert_allclose(result[1], [[28.8]], atol=1e-6)
        for original, (actual, _) in zip(before, submissions):
            for left, right in zip(original, actual):
                np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(global_parameters[0], [10, -5])

    def test_clipping_sample_weights_and_server_relative_delta(self):
        global_parameters = [np.array([100.0], dtype=np.float64)]
        submissions = [
            ([np.array([100.0 + d])], count)
            for d, count in [(1, 1), (2, 3), (100, 1)]
        ]
        result, logs = aggregate_updates(
            global_parameters, submissions, "clip_fedavg"
        )
        self.assertEqual(logs[0]["threshold"], 2)
        np.testing.assert_allclose(result[0], [101.8])

    def test_zero_threshold_zero_updates_and_determinism(self):
        initial = [np.array([7.0], dtype=np.float32)]
        submissions = [
            ([np.array([v], dtype=np.float32)], 1) for v in [7, 7, 100]
        ]
        result, logs = aggregate_updates(initial, submissions, "clip_fedavg")
        repeated, repeated_logs = aggregate_updates(
            initial, submissions, "clip_fedavg"
        )
        np.testing.assert_array_equal(result[0], initial[0])
        np.testing.assert_array_equal(result[0], repeated[0])
        self.assertEqual(logs, repeated_logs)
        self.assertEqual([r["clip_factor"] for r in logs], [1, 1, 0])

    def test_median_shapes_dtypes_and_coordinate_values(self):
        for dtype in (np.float32, np.float64):
            for count in (5, 7):
                initial = [
                    np.full((2, 3), 10, dtype=dtype),
                    np.zeros(2, dtype=dtype),
                ]
                submissions = [
                    (
                        [
                            initial[0]
                            + np.array([[i, -i, i]] * 2, dtype=dtype),
                            np.array([i, -i], dtype=dtype),
                        ],
                        1 if i else 10000,
                    )
                    for i in range(count)
                ]
                result, _ = aggregate_updates(
                    initial, submissions, "coordinate_median"
                )
                middle = count // 2
                for reference, value in zip(initial, result):
                    self.assertEqual(value.shape, reference.shape)
                    self.assertEqual(value.dtype, reference.dtype)
                np.testing.assert_array_equal(
                    result[0], initial[0] + [[middle, -middle, middle]] * 2
                )
                np.testing.assert_array_equal(result[1], [middle, -middle])
                reordered, _ = aggregate_updates(
                    initial, list(reversed(submissions)), "coordinate_median"
                )
                for left, right in zip(result, reordered):
                    np.testing.assert_array_equal(left, right)

    def test_three_large_sybil_updates_are_clipped(self):
        initial = [np.zeros(1, dtype=np.float32)]
        submissions = [
            ([np.array([value], dtype=np.float32)], 12000)
            for value in (1, 2, 3, 4, 100000, 100000, 100000)
        ]
        result, logs = aggregate_updates(initial, submissions, "clip_fedavg")
        np.testing.assert_allclose(result[0], [22 / 7], rtol=1e-6)
        for log in logs[-3:]:
            self.assertEqual(log["threshold"], 4)
            self.assertLessEqual(log["clipped_update_l2"], 4)
            self.assertLessEqual(log["reconstructed_update_l2"], 4)
        median, _ = aggregate_updates(
            initial, submissions, "coordinate_median"
        )
        np.testing.assert_array_equal(median[0], [4])

    def test_fedavg_exactly_matches_existing_aggregate(self):
        initial = [np.array([2, 5], dtype=np.float32)]
        submissions = [
            ([np.array([i, i + 1], dtype=np.float32)], i + 1) for i in range(5)
        ]
        expected = aggregate([([p[0].copy()], n) for p, n in submissions])
        actual, _ = aggregate_updates(initial, submissions, "fedavg")
        np.testing.assert_array_equal(actual[0], expected[0])

    def test_rejects_bad_shapes_dtypes_nonfinite_and_counts(self):
        initial = [np.zeros(2, dtype=np.float32)]
        bad = [
            ([np.zeros(3, dtype=np.float32)], 1),
            ([np.zeros(2, dtype=np.float64)], 1),
            ([np.array([np.nan, 0], dtype=np.float32)], 1),
            ([np.zeros(2, dtype=np.float32)], 0),
            ([], 1),
        ]
        for submission in bad:
            with self.assertRaises(ValueError):
                aggregate_updates(initial, [submission], "clip_fedavg")
        with self.assertRaises(ValueError):
            aggregate_updates(initial, [], "coordinate_median")


class PilotTests(unittest.TestCase):
    def test_matched_benchmarks_and_shared_sybil_data(self):
        labels = torch.arange(100) % 10
        dataset = TensorDataset(torch.zeros(100, 1, 28, 28), labels)
        dataset.targets = labels
        pair = dataset, dataset
        parts = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]
        for benchmark in pilot.BENCHMARKS:
            clean, clean_parts, _ = pilot.make_workers(
                benchmark, pair, parts, 42, False
            )
            attack, attack_parts, _ = pilot.make_workers(
                benchmark, pair, parts, 42, True
            )
            self.assertEqual(
                pilot.backdoor.identity(clean, clean_parts),
                pilot.backdoor.identity(attack, attack_parts),
            )
            for left, right in zip(clean, attack):
                self.assertEqual(
                    list(left.trainloader.sampler),
                    list(right.trainloader.sampler),
                )
            if benchmark == "model_poisoning":
                self.assertFalse(
                    any(
                        isinstance(w.trainset, pilot.backdoor.TriggeredDataset)
                        for w in attack
                    )
                )
            if benchmark == "sybil_backdoor":
                self.assertEqual(len(attack), 7)
                self.assertEqual(attack_parts[0], attack_parts[5])
                self.assertEqual(attack_parts[0], attack_parts[6])

    def test_model_submission_reuses_fit_scaling_and_restores_globals(self):
        before = pilot.client.ENABLE_POISON
        worker = Mock()

        def fake_fit(parameters, config):
            self.assertTrue(pilot.client.ENABLE_POISON)
            self.assertEqual(pilot.client.POISON_SCALE, 10)
            return parameters, 12000, {}

        worker.fit.side_effect = fake_fit
        initial = [np.array([1], dtype=np.float32)]
        result = pilot.local_submissions(
            [worker], initial, "model_poisoning", True
        )
        self.assertEqual(result[0][1], 12000)
        self.assertEqual(pilot.client.ENABLE_POISON, before)

    def test_summary_pairs_clean_cost_with_own_fedavg_control(self):
        final = {}
        for benchmark in pilot.BENCHMARKS:
            for method in pilot.METHODS:
                for offset, seed in enumerate(pilot.SEEDS):
                    for attack in (False, True):
                        final[benchmark, method, seed, attack] = {
                            "clean_accuracy_percent": 90
                            - (offset if method != "fedavg" else 0)
                            - (2 if attack else 0),
                            "clean_loss": 0.5
                            + (0.1 if method != "fedavg" else 0)
                            + (0.2 if attack else 0),
                            "asr_percent": (
                                None
                                if benchmark == "model_poisoning"
                                else (60 if attack else 1)
                            ),
                        }
        rows = pilot.paired_rows(final)
        self.assertEqual(len(rows), 27)
        summary = pilot.summarize(rows)
        self.assertEqual(len(summary), 9)
        for row in summary:
            self.assertEqual(row["accuracy_damage_pp_mean"], 2)
            if row["method"] != "fedavg":
                self.assertEqual(
                    row["defense_clean_accuracy_degradation_pp_mean"], 1
                )
                self.assertEqual(
                    row["defense_clean_accuracy_degradation_pp_sample_sd"], 1
                )
            if row["benchmark"] == "model_poisoning":
                self.assertIsNone(row["paired_asr_increase_pp_mean"])
            else:
                self.assertEqual(row["paired_asr_increase_pp_mean"], 59)
        with self.assertRaises(ValueError):
            pilot.summarize(rows[:-1])

    def test_round_loop_with_mock_submissions_no_training(self):
        worker = Mock()
        worker.get_parameters.return_value = [np.zeros(2, dtype=np.float32)]
        submissions = [
            ([np.ones(2, dtype=np.float32) * i], 12000) for i in (1, 2, 100)
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                pilot, "local_submissions", return_value=submissions
            ), patch.object(
                pilot.client, "test", return_value=(0.1, 0.9)
            ), patch.object(
                pilot.client, "train", side_effect=AssertionError
            ):
                final = pilot.train_condition(
                    [worker],
                    "model_poisoning",
                    "clip_fedavg",
                    True,
                    Path(directory),
                    None,
                )
            self.assertEqual(final["clean_accuracy_percent"], 90)
            self.assertIsNone(final["asr_percent"])
            self.assertEqual(
                len(
                    pilot.experiment_c.read_rows(
                        Path(directory) / "results.csv"
                    )
                ),
                10,
            )
            self.assertEqual(
                len(
                    pilot.experiment_c.read_rows(
                        Path(directory) / "aggregation.csv"
                    )
                ),
                30,
            )

    def test_incomplete_or_changed_archive_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(FileNotFoundError):
                pilot.verify_archive(folder)
            (folder / "data").write_text("original")
            pilot.backdoor.write_json(
                folder / "checksums.json", pilot.sweep.inventory(folder)
            )
            pilot.verify_archive(folder)
            (folder / "data").write_text("changed")
            with self.assertRaises(ValueError):
                pilot.verify_archive(folder)

    def test_source_compatibility_rejects_logic_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            current = Path("client.py").read_text()
            saved = root / "source/client.py"
            environment = {
                "python": pilot.platform.python_version(),
                "packages": {
                    name: pilot.importlib.metadata.version(name)
                    for name in ("torch", "torchvision", "numpy", "flwr")
                },
                "torch_threads": 1,
                "deterministic_algorithms": True,
            }
            for changed, should_pass in (
                (current + "\n# Formatting-only provenance test\n", True),
                (current.replace("lr=0.01", "lr=0.02"), False),
            ):
                self.assertNotEqual(changed, current)
                saved.write_text(changed)
                environment["source_sha256"] = {
                    "client.py": pilot.sweep.sha256(saved)
                }
                pilot.backdoor.write_json(
                    root / "environment.json", environment
                )
                if should_pass:
                    result = pilot.verify_environment(root, ("client.py",))
                    self.assertEqual(
                        result["client.py"]["match"], "identical Python AST"
                    )
                    with self.assertRaises(ValueError):
                        pilot.verify_environment(root)
                else:
                    with self.assertRaises(ValueError):
                        pilot.verify_environment(root, ("client.py",))

    def test_preview_and_existing_output_never_train(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            with patch.object(
                pilot.sys, "argv", ["defense", "--output", str(output)]
            ), patch.object(
                pilot.sweep, "current_data", return_value=(None, None)
            ), patch.object(
                pilot, "preflight", return_value=({}, {}, {}, {})
            ), patch.object(
                pilot, "execute", side_effect=AssertionError
            ):
                pilot.main()
            self.assertFalse(output.exists())
            with patch.object(
                pilot.sys, "argv", ["defense", "--run", "--output", directory]
            ), patch.object(
                pilot.sweep, "current_data", side_effect=AssertionError
            ):
                with self.assertRaises(SystemExit):
                    pilot.main()


if __name__ == "__main__":
    unittest.main()
