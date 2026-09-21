"""快速檢查實驗 D 的聚合公式，不啟動 MNIST 訓練。"""

import unittest

import numpy as np

from run_weight_comparison import combine, equal_average


class WeightComparisonTests(unittest.TestCase):
    def test_equal_average_ignores_client_sample_count(self):
        results = [
            ([np.array([float(cid)])], 1 if cid == 0 else 9)
            for cid in range(5)
        ]
        averaged = equal_average(results)
        np.testing.assert_array_equal(averaged[0], np.array([2.0]))

    def test_weighted_and_equal_averages_differ_when_counts_differ(self):
        results = [
            ([np.array([0.0])], 1),
            ([np.array([1.0])], 9),
        ]
        equal = combine(results, "equal")[0]
        weighted = combine(results, "weighted")[0]
        self.assertAlmostEqual(float(equal.item()), 0.5)
        self.assertAlmostEqual(float(weighted.item()), 0.9)


if __name__ == "__main__":
    unittest.main()
