"""Experimental near-duplicate grouping; aggregation never receives labels."""

import numpy as np

from defense_aggregation import aggregate_updates
from run_update_similarity import update_statistics

METHOD = "clip_similarity_group"
COSINE_THRESHOLD = 0.999999
DEFINITION = {
    "clipping": "unchanged validated median-norm clipping",
    "cosine_threshold": COSINE_THRESHOLD,
    "directions": (
        "submitted float64 deltas before clipping; zero is undefined"
    ),
    "edges": "cosine >= 0.999999; zero updates have no edges",
    "groups": "connected components; sorted members, ascending smallest ID",
    "within_group": "unweighted arithmetic mean of clipped contributions",
    "between_groups": "unweighted arithmetic mean; one vote per component",
    "rounding": (
        "native-dtype clipped parameters, float64 means, native output"
    ),
    "scope": (
        "equal reported sample counts; identical-replica Sybil pilot only"
    ),
    "threshold_selection": "fixed near-identity tolerance, no outcome tuning",
}


def connected_components(count, pairs):
    """Undirected components; transitive chains are deliberately one group."""
    neighbors = [set() for _ in range(count)]
    for i, j, cosine in pairs:
        if cosine is not None and cosine >= COSINE_THRESHOLD:
            neighbors[i].add(j)
            neighbors[j].add(i)
    groups = []
    visited = set()
    for start in range(count):
        if start in visited:
            continue
        pending = [start]
        group = []
        visited.add(start)
        while pending:
            node = pending.pop()
            group.append(node)
            for neighbor in sorted(neighbors[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        groups.append(sorted(group))
    return groups


def aggregate_grouped(global_parameters, submissions):
    """Return parameters, clipping audit, components and unlabeled cosines."""
    if not submissions or len({count for _, count in submissions}) != 1:
        raise ValueError("This pilot requires equal positive sample counts")
    # Reuse all validated clipping checks, tau, and per-client clip factors.
    _, records = aggregate_updates(
        global_parameters, submissions, "clip_fedavg"
    )
    _, pairs = update_statistics(global_parameters, submissions)
    groups = connected_components(len(submissions), pairs)
    result = []
    for index, reference in enumerate(global_parameters):
        clipped = []
        for (parameters, _), record in zip(submissions, records):
            delta = parameters[index].astype(np.float64) - reference.astype(
                np.float64
            )
            reconstructed = (
                reference.astype(np.float64) + record["clip_factor"] * delta
            ).astype(reference.dtype)
            clipped.append(reconstructed.astype(np.float64))
        contributions = [
            np.mean([clipped[i] for i in group], axis=0) for group in groups
        ]
        value = np.mean(contributions, axis=0).astype(reference.dtype)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite grouped aggregation")
        result.append(value)
    return result, records, groups, pairs


def analyze_groups(groups, replica_ids):
    """Post-aggregation weights exclude clipping attenuation."""
    replicas = set(replica_ids)
    weight = 0.0
    honest_grouped = 0
    honest_pairs = 0
    mixed = 0
    for group in groups:
        replica_count = len(replicas.intersection(group))
        honest_count = len(group) - replica_count
        weight += replica_count / len(group) / len(groups)
        honest_pairs += honest_count * (honest_count - 1) // 2
        if len(group) > 1:
            honest_grouped += honest_count
        mixed += int(replica_count > 0 and honest_count > 0)
    return {
        "effective_participants": len(groups),
        "max_group_size": max(map(len, groups)),
        "effective_attacker_coalition_weight": weight,
        "honest_identities_in_nonsingleton_groups": honest_grouped,
        "honest_honest_pairs_grouped": honest_pairs,
        "mixed_groups": mixed,
    }
