import csv
from pathlib import Path
import tempfile
import unittest

from run_study import SEEDS, key, settings, write_summary


class StudyTests(unittest.TestCase):
    def test_design_varies_one_setting_and_shares_overlap(self):
        a = settings('A')
        b = settings('B')
        self.assertEqual([r[1] for r in a], [0.01, 0.1, 1, 10, 100])
        self.assertTrue(all(r[2] == 10 for r in a))
        self.assertEqual([r[2] for r in b], [1, 2, 5, 10, 20])
        self.assertTrue(all(r[1] == 0.1 for r in b))
        unique = {key(alpha, scale) for _, alpha, scale in settings('all')}
        self.assertEqual(len(unique), 9)
        self.assertEqual((len(unique) - 2) * len(SEEDS) * 2, 42)

    def test_damage_is_paired_and_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / 'experiment'
            for seed, attack_acc, attack_loss in zip(SEEDS, [0.7, 0.9, 0.5], [2, 3, 4]):
                for mode, accuracy, loss in [('clean', 0.8, 1), ('poisoned', attack_acc, attack_loss)]:
                    folder = experiment / f'seed_{seed}' / mode
                    folder.mkdir(parents=True)
                    (folder / 'results.csv').write_text(f'round,loss,accuracy\n10,{loss},{accuracy}\n')
            write_summary(root, [dict(experiment='A', alpha=0.1, scale=10, path=str(experiment))])
            with (root / 'summary.csv').open() as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row['accuracy_damage_pp_mean']), 10)
            self.assertAlmostEqual(float(row['accuracy_damage_pp_sample_std']), 20)
            self.assertAlmostEqual(float(row['loss_damage_mean']), 2)
            self.assertAlmostEqual(float(row['loss_damage_sample_std']), 1)


if __name__ == '__main__':
    unittest.main()
