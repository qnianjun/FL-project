"""Quick sweep checks; no training or downloads."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_backdoor_sweep as sweep


class SweepTests(unittest.TestCase):
    def test_training_dispatch_changes_only_alpha_and_restores_settings(self):
        original = sweep.pilot.SETTINGS

        def inspect_dispatch():
            actual = sweep.asdict(sweep.pilot.SETTINGS)
            expected = sweep.asdict(original)
            expected["alpha"] = 100
            self.assertEqual(actual, expected)
            self.assertEqual(sweep.sys.argv, [
                "run_backdoor.py", "--run", "--output", "unused",
            ])
            raise RuntimeError("Simulated interruption; no training")

        with patch.object(sweep.pilot, "main", side_effect=inspect_dispatch):
            with self.assertRaises(RuntimeError):
                sweep.train_alpha(100, Path("unused"))
        self.assertIs(sweep.pilot.SETTINGS, original)

    def test_paired_differences_precede_sample_sd(self):
        rows = []
        for alpha in sweep.ALPHAS:
            final = []
            for index, seed in enumerate((42, 43, 44)):
                for mode in ("clean", "backdoor"):
                    attack = mode == "backdoor"
                    final.append({
                        "seed": seed, "condition": mode,
                        "clean_accuracy_percent": (
                            90 + index - (index + 1 if attack else 0)
                        ),
                        "asr_percent": 5 + index + (2 * index if attack else 0),
                    })
            rows.extend(sweep.paired_rows(alpha, list(reversed(final))))
        summary = sweep.summary_rows(rows)
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(summary), 5)
        for row in summary:
            self.assertEqual(row["paired_clean_accuracy_change_pp_mean"], -2)
            self.assertEqual(row["paired_clean_accuracy_change_pp_sample_sd"], 1)
            self.assertEqual(row["paired_asr_increase_pp_mean"], 2)
            self.assertEqual(row["paired_asr_increase_pp_sample_sd"], 2)
        with self.assertRaises(ValueError):
            sweep.summary_rows(rows[:-1])

    def test_existing_output_is_never_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep.require_fresh_output(root / "fresh")
            with self.assertRaises(ValueError):
                sweep.require_fresh_output(root)
            with self.assertRaises(FileNotFoundError):
                sweep.verify_checksums(root)
            path = root / "summary.csv"
            path.write_text("existing\n")
            with self.assertRaises(FileExistsError):
                sweep.write_csv(path, [{"alpha": 1}])
            self.assertEqual(path.read_text(), "existing\n")

    def test_metrics_reject_partial_nonfinite_and_bad_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            rows = [[number, 0.1, 90, 5] for number in range(1, 11)]

            def write(values):
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["round", *sweep.METRICS])
                    writer.writerows(values)

            write(rows)
            self.assertEqual(sweep.read_metrics(path)["round"], 10)
            write(rows[:-1])
            with self.assertRaises(ValueError):
                sweep.read_metrics(path)
            rows[4][0] = 4
            write(rows)
            with self.assertRaises(ValueError):
                sweep.read_metrics(path)
            rows[4] = [5, 0.1, float("nan"), 5]
            write(rows)
            with self.assertRaises(ValueError):
                sweep.read_metrics(path)

    def test_checksums_reject_tampering_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in sweep.expected_files():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            sweep.pilot.write_json(root / "checksums.json", sweep.inventory(root))
            sweep.verify_checksums(root)
            (root / "unexpected").write_text("extra")
            with self.assertRaises(ValueError):
                sweep.verify_checksums(root)
            (root / "unexpected").unlink()
            (root / "summary.json").write_text("tampered")
            with self.assertRaises(ValueError):
                sweep.verify_checksums(root)


if __name__ == "__main__":
    unittest.main()
