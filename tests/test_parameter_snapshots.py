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
