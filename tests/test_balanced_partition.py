import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset
from balanced_partition import balanced_partition
from run_balanced import BalancedClient
import client


class BalancedPartitionTests(unittest.TestCase):
    def test_complete_balanced_reproducible_and_seed_changes(self):
        labels=np.tile(np.arange(10),100)
        for alpha in [0.01,0.1,1,10,100]:
            a=balanced_partition(labels,5,alpha,42)
            self.assertEqual([len(p) for p in a],[200]*5)
            self.assertEqual(sorted(i for p in a for i in p),list(range(1000)))
            self.assertEqual(a,balanced_partition(labels,5,alpha,42))
            self.assertNotEqual(a,balanced_partition(labels,5,alpha,43))
    def test_capacity_fallback_for_tiny_alpha(self):
        parts=balanced_partition(np.zeros(100,dtype=int),5,0.001,42)
        self.assertEqual([len(p) for p in parts],[20]*5)
    def test_rejects_bad_input(self):
        for alpha in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):balanced_partition(np.arange(100),5,alpha,42)
        with self.assertRaises(ValueError):balanced_partition(np.arange(101),5,1,42)
    def test_small_training_pair_and_attack(self):
        torch.manual_seed(42)
        dataset=TensorDataset(torch.rand(100,1,28,28),torch.arange(100)%10)
        parts=balanced_partition(np.arange(100)%10,5,0.1,42)
        a=BalancedClient(0,(dataset,dataset),parts,42)
        b=BalancedClient(0,(dataset,dataset),parts,42)
        initial=a.get_parameters({})
        with patch.object(client,'ENABLE_POISON',False):clean,_,_=a.fit(initial,{})
        with patch.multiple(client,ENABLE_POISON=True,POISON_CLIENT=0,POISON_SCALE=10):poisoned,_,_=b.fit(initial,{})
        self.assertTrue(any(np.any(c!=i) for c,i in zip(clean,initial)))
        for i,c,p in zip(initial,clean,poisoned):
            np.testing.assert_allclose(p,i+10*(c-i),rtol=1e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
