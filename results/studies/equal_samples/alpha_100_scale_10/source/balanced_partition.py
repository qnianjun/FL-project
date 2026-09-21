"""Capacity-constrained Dirichlet partitioning for experiment C.

Draw one client preference vector per label. Visit every sample in shuffled order,
choose a client using its label preference among clients with free slots, and stop
assigning to a client once full. This is a constrained variant, not the original
unconstrained Dirichlet partition. Realized label distributions must be inspected.
"""
import numpy as np


def balanced_partition(targets, num_clients, alpha, seed):
    labels = np.asarray(targets)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError('targets must be a nonempty one-dimensional array')
    if num_clients < 1 or len(labels) % num_clients:
        raise ValueError('Sample count must be divisible by num_clients')
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    rng = np.random.default_rng(seed)
    classes, encoded = np.unique(labels, return_inverse=True)
    preferences = rng.dirichlet(np.full(num_clients, alpha), size=len(classes))
    capacity = np.full(num_clients, len(labels) // num_clients, dtype=int)
    partitions = [[] for _ in range(num_clients)]
    for index in rng.permutation(len(labels)):
        weights = preferences[encoded[index]].copy()
        weights[capacity == 0] = 0
        # Tiny-alpha draws can give zero probability to all remaining clients.
        if weights.sum() == 0:
            weights = (capacity > 0).astype(float)
        weights /= weights.sum()
        cid = int(rng.choice(num_clients, p=weights))
        partitions[cid].append(int(index))
        capacity[cid] -= 1
    assert np.all(capacity == 0)
    return partitions
