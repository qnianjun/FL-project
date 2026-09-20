import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import client


class ParameterSnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Exercise real training without downloading MNIST or starting Flower.
        self.worker = client.FlowerClient.__new__(client.FlowerClient)
        self.worker.cid = 0
        self.worker.model = client.Net()
        self.worker.trainset = TensorDataset(
            torch.rand(8, 1, 28, 28), torch.arange(8)
        )
        self.worker.trainloader = DataLoader(self.worker.trainset, batch_size=8)

    def test_snapshot_survives_training(self):
        before = self.worker.get_parameters({})
        expected = [value.copy() for value in before]
        client.train(self.worker.model, self.worker.trainloader)
        after = self.worker.get_parameters({})

        for snapshot, original in zip(before, expected):
            np.testing.assert_array_equal(snapshot, original)
        self.assertTrue(any(np.any(a != b) for a, b in zip(after, before)))

    def test_seed_reproduces_partition_initialization_and_training(self):
        labels = torch.arange(10).repeat(20)
        dataset = TensorDataset(torch.rand(200, 1, 28, 28), labels)
        dataset.targets = labels
        with patch.multiple(client, SEED=43, ALPHA=1, ENABLE_POISON=False):
            first = client.FlowerClient(0, (dataset, dataset))
            initial = first.get_parameters({})
            trained, _, _ = first.fit(initial, {})
            torch.rand(100)  # Unrelated global RNG use must not affect replay.
            repeated = client.FlowerClient(0, (dataset, dataset))
            self.assertEqual(first.trainset.indices, repeated.trainset.indices)
            for left, right in zip(initial, repeated.get_parameters({})):
                np.testing.assert_array_equal(left, right)
            replayed, _, _ = repeated.fit(initial, {})
            for left, right in zip(trained, replayed):
                np.testing.assert_array_equal(left, right)

    def test_partition_keeps_samples_when_proportions_round_down(self):
        dataset = TensorDataset(torch.zeros(10, 1), torch.arange(10))
        dataset.targets = torch.arange(10)
        proportions = np.array([0.2, 0.2, 0.2, 0.2, 0.2 - 1e-15])
        with patch.object(client.np.random, "dirichlet", return_value=proportions):
            partitions = client.create_dirichlet_partition(dataset, 5, 0.1)
        self.assertEqual(sorted(i for group in partitions for i in group), list(range(10)))

    def test_poisoning_scales_real_training_update(self):
        initial = self.worker.get_parameters({})
        with patch.object(client, "ENABLE_POISON", False):
            clean, count, _ = self.worker.fit(initial, {})
        self.assertEqual(count, 8)
        self.assertTrue(any(np.any(a != b) for a, b in zip(clean, initial)))

        for scale in (1, 10, 100):
            with self.subTest(scale=scale), patch.multiple(
                client, ENABLE_POISON=True, POISON_CLIENT=0, POISON_SCALE=scale
            ):
                poisoned, _, _ = self.worker.fit(initial, {})
                for start, normal, attacked in zip(initial, clean, poisoned):
                    np.testing.assert_allclose(
                        attacked, start + scale * (normal - start),
                        rtol=1e-5, atol=1e-7,
                    )

        with patch.multiple(
            client, ENABLE_POISON=True, POISON_CLIENT=1, POISON_SCALE=100
        ):
            honest, _, _ = self.worker.fit(initial, {})
        for expected, actual in zip(clean, honest):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
