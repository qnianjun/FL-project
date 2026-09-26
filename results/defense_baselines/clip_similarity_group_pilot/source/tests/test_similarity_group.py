"""Synthetic grouping and orchestration checks; no FL training."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from defense_aggregation import aggregate_updates
import run_similarity_group as stage
from similarity_group_aggregation import (
    COSINE_THRESHOLD,
    aggregate_grouped,
    analyze_groups,
    connected_components,
)


class GroupingTests(unittest.TestCase):
    def submissions(self):
        return [
            ([np.array(value, dtype=np.float32)], 12000)
            for value in (
                [10, 0],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 1],
                [10, 0],
                [10, 0],
            )
        ]

    def test_one_group_one_vote_and_validated_clipping(self):
        initial = [np.zeros(2, dtype=np.float32)]
        submissions = self.submissions()
        before = [p[0].copy() for p, _ in submissions]
        result, logs, groups, _ = aggregate_grouped(initial, submissions)
        _, expected_logs = aggregate_updates(
            initial, submissions, "clip_fedavg"
        )
        self.assertEqual(logs, expected_logs)
        self.assertEqual(groups, [[0, 5, 6], [1], [2], [3], [4]])
        np.testing.assert_allclose(result[0], [np.sqrt(2) / 5, 0.2], atol=1e-7)
        self.assertEqual(result[0].shape, initial[0].shape)
        self.assertEqual(result[0].dtype, initial[0].dtype)
        for old, (p, _) in zip(before, submissions):
            np.testing.assert_array_equal(old, p[0])
        for log in logs:
            self.assertLessEqual(
                log["clipped_update_l2"], log["threshold"] + 1e-12
            )
        analysis = analyze_groups(groups, {0, 5, 6})
        self.assertAlmostEqual(
            analysis["effective_attacker_coalition_weight"], 0.2
        )
        self.assertEqual(
            analysis["honest_identities_in_nonsingleton_groups"], 0
        )

    def test_components_transitive_deterministic_and_boundary(self):
        edges = [
            (2, 3, COSINE_THRESHOLD),
            (0, 2, 1.0),
            (0, 3, 0.99),
            (1, 4, None),
            (1, 2, COSINE_THRESHOLD - 1e-8),
        ]
        expected = [[0, 2, 3], [1], [4]]
        self.assertEqual(connected_components(5, edges), expected)
        self.assertEqual(connected_components(5, edges[::-1]), expected)
        self.assertEqual(connected_components(5, edges), expected)

    def test_zero_updates_singletons_and_dtype_shapes(self):
        initial = [np.ones((2, 3), dtype=np.float64), np.ones(3)]
        submissions = [([v.copy() for v in initial], 12000) for _ in range(7)]
        result, _, groups, pairs = aggregate_grouped(initial, submissions)
        self.assertEqual(groups, [[i] for i in range(7)])
        self.assertTrue(all(value is None for _, _, value in pairs))
        for left, right in zip(initial, result):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(left.dtype, right.dtype)

    def test_no_label_input_and_permutation_equivariance(self):
        initial = [np.zeros(2, dtype=np.float32)]
        submissions = self.submissions()
        left, _, groups, _ = aggregate_grouped(initial, submissions)
        analyze_groups(groups, {1, 2, 3})
        repeated, _, _, _ = aggregate_grouped(initial, submissions)
        np.testing.assert_array_equal(left[0], repeated[0])
        permuted = [submissions[i] for i in [6, 3, 1, 5, 2, 0, 4]]
        right, _, _, _ = aggregate_grouped(initial, permuted)
        np.testing.assert_array_equal(left[0], right[0])
        with self.assertRaises(TypeError):
            aggregate_grouped(initial, submissions, replica_ids={0, 5, 6})

    def test_false_grouping_accounting_and_unequal_counts_rejected(self):
        analysis = analyze_groups([[0, 1, 2], [3, 4], [5, 6]], {0, 5, 6})
        self.assertEqual(
            analysis["honest_identities_in_nonsingleton_groups"], 4
        )
        self.assertEqual(analysis["honest_honest_pairs_grouped"], 2)
        self.assertEqual(analysis["mixed_groups"], 1)
        self.assertAlmostEqual(
            analysis["effective_attacker_coalition_weight"], 4 / 9
        )
        submissions = self.submissions()
        submissions[-1] = (submissions[-1][0], 1)
        with self.assertRaises(ValueError):
            aggregate_grouped([np.zeros(2, dtype=np.float32)], submissions)

    def test_real_defense_loop_writes_round_diagnostics_without_training(self):
        class Worker:
            model = None
            testloader = None

            def get_parameters(self, config):
                return [np.zeros(2, dtype=np.float32)]

            def set_parameters(self, parameters):
                self.parameters = parameters

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with (
                patch.object(
                    stage.baseline,
                    "local_submissions",
                    return_value=self.submissions(),
                ),
                patch.object(
                    stage.baseline.client, "train", side_effect=AssertionError
                ),
                patch.object(
                    stage.baseline.client, "test", return_value=(0.5, 0.8)
                ),
            ):
                metrics, _ = stage.train_condition(
                    [Worker()], True, folder, None
                )
            self.assertEqual(metrics["asr_percent"], 80)
            self.assertEqual(
                len(stage.read_rows(folder / "aggregation.csv")), 70
            )
            self.assertEqual(
                len(stage.read_rows(folder / "pairwise_similarity.csv")), 210
            )
            self.assertEqual(
                len(stage.read_rows(folder / "grouping_per_round.csv")), 10
            )
            self.assertEqual(len(stage.read_rows(folder / "results.csv")), 10)
            first = stage.read_rows(folder / "groups.csv")[0]
            self.assertEqual(first["members"], "[0, 5, 6]")

    def test_preview_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "fresh"
            with (
                patch.object(
                    stage.sys, "argv", ["stage", "--output", str(fresh)]
                ),
                patch.object(
                    stage.sweep, "current_data", return_value=(None, None)
                ),
                patch.object(
                    stage, "preflight", return_value=({}, {}, {}, {}, {})
                ),
                patch.object(stage, "execute", side_effect=AssertionError),
            ):
                stage.main()
            self.assertFalse(fresh.exists())
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
                with self.assertRaises(SystemExit):
                    stage.main()

    def test_comparison_uses_paired_seeds_and_sample_sd(self):
        final = {}
        for method in stage.METHODS:
            for index, seed in enumerate((42, 43, 44)):
                for attack in (False, True):
                    final[stage.BENCHMARK, method, seed, attack] = {
                        "clean_accuracy_percent": (
                            90 - (index + 1 if method == stage.METHOD else 0)
                        ),
                        "clean_loss": 0.5,
                        "asr_percent": 1 + (2 * index if attack else 0),
                    }
        rows, summary = stage.comparison_rows(final)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(summary), 4)
        record = next(r for r in summary if r["method"] == stage.METHOD)
        self.assertEqual(record["paired_asr_increase_pp_mean"], 2)
        self.assertEqual(record["paired_asr_increase_pp_sample_sd"], 2)
        self.assertEqual(
            record["defense_clean_accuracy_degradation_pp_mean"], 2
        )
        self.assertEqual(
            record["defense_clean_accuracy_degradation_pp_sample_sd"], 1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_seed.csv"
            stage.sweep.write_csv(path, rows)
            stage.validate_table(path, rows)
            path.write_text(path.read_text().replace("90", "89", 1))
            with self.assertRaises(ValueError):
                stage.validate_table(path, rows)

    def test_incomplete_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "checksums.json").write_text(
                '{"results.csv": "missing"}'
            )
            with self.assertRaises(ValueError):
                stage.baseline.verify_archive(folder)


if __name__ == "__main__":
    unittest.main()
