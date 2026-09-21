"""Synthetic-data checks: no downloads or FL training."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import client
from run_backdoor import (
    SETTINGS,
    TriggeredDataset,
    apply_trigger,
    asr_dataset,
    identity,
    make_workers,
    poison_selection,
    summarize,
    verify_transforms,
)


class BackdoorTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.zeros(100, 1, 28, 28)
        self.labels = torch.arange(100) % 10
        self.dataset = TensorDataset(self.images, self.labels)
        self.dataset.targets = self.labels

    def test_trigger_placement_and_no_mutation(self):
        image = self.images[0]
        triggered = apply_trigger(image)
        expected = torch.zeros_like(image)
        expected[:, 25:28, 25:28] = 1
        self.assertTrue(torch.equal(triggered, expected))
        self.assertEqual(torch.count_nonzero(image).item(), 0)
        self.assertNotEqual(image.data_ptr(), triggered.data_ptr())

    def test_relabeling_only_selected_samples(self):
        wrapped = TriggeredDataset(self.dataset, [1, 4])
        for index in range(len(wrapped)):
            image, label = wrapped[index]
            if index in (1, 4):
                self.assertEqual(label, SETTINGS.target_label)
                self.assertEqual(torch.count_nonzero(image).item(), 9)
            else:
                self.assertEqual(label, self.labels[index])
                self.assertEqual(torch.count_nonzero(image).item(), 0)
        self.assertTrue(torch.equal(self.labels, torch.arange(100) % 10))
        self.assertEqual(torch.count_nonzero(self.images).item(), 0)

    def test_selection_deterministic_and_fixed_count(self):
        selected = poison_selection(101, 0.2, 2042)
        np.random.seed(123)
        np.random.random(20)
        np.testing.assert_array_equal(selected, poison_selection(101, 0.2, 2042))
        self.assertEqual(len(set(selected)), 20)
        self.assertTrue(np.all((selected >= 0) & (selected < 101)))
        self.assertFalse(np.array_equal(selected, poison_selection(101, 0.2, 2043)))
        self.assertEqual(len(poison_selection(100, 0, 42)), 0)
        self.assertEqual(len(poison_selection(100, 1, 42)), 100)
        with self.assertRaises(ValueError):
            poison_selection(100, -0.1, 42)

    def test_matched_workers_shuffles_and_attacker_isolation(self):
        partitions = [list(range(cid * 20, (cid + 1) * 20)) for cid in range(5)]
        pair = (self.dataset, self.dataset)
        clean, selected = make_workers(pair, partitions, 42, False)
        backdoor, repeated = make_workers(pair, partitions, 42, True)
        self.assertEqual(identity(clean, partitions), identity(backdoor, partitions))
        np.testing.assert_array_equal(selected, repeated)
        verify_transforms(clean, pair, selected, False)
        verify_transforms(backdoor, pair, selected, True)
        for left, right in zip(clean, backdoor):
            for _ in range(2):
                self.assertEqual(
                    list(left.trainloader.sampler), list(right.trainloader.sampler)
                )
        self.assertEqual(torch.count_nonzero(self.images).item(), 0)

    def test_asr_excludes_original_targets_and_preserves_clean_test(self):
        triggered = asr_dataset(self.dataset)
        self.assertEqual(len(triggered), 90)
        self.assertTrue(all(self.labels[i] != 0 for i in triggered.dataset.indices))
        self.assertTrue(all(triggered[i][1] == 0 for i in range(len(triggered))))
        # A constant target prediction must have 100% ASR and 10% clean accuracy.
        model = client.Net()
        logits = torch.zeros(1, 10)
        logits[0, 0] = 1
        with patch.object(model, "forward", side_effect=lambda x: logits.repeat(len(x), 1)):
            _, asr = client.test(model, DataLoader(triggered, batch_size=32))
            _, clean = client.test(model, DataLoader(self.dataset, batch_size=32))
        self.assertEqual(asr, 1)
        self.assertEqual(clean, 0.1)
        self.assertEqual(torch.count_nonzero(self.images).item(), 0)

    def test_paired_summary_and_sample_standard_deviation(self):
        rows = []
        for index, seed in enumerate(SETTINGS.seeds):
            for mode, offset in (("clean", 0), ("backdoor", 2)):
                rows.append({
                    "seed": seed, "condition": mode,
                    "clean_loss": index + offset,
                    "clean_accuracy_percent": index + offset,
                    "asr_percent": index + offset,
                })
        summary = summarize(list(reversed(rows)))
        stats = summary["conditions"]["clean"]["asr_percent"]
        self.assertEqual(stats, {"mean": 1, "sample_std": 1})
        paired = summary["paired_backdoor_minus_clean"]["statistics"]
        self.assertEqual(paired["asr_percent"], {"mean": 2, "sample_std": 0})


if __name__ == "__main__":
    unittest.main()
