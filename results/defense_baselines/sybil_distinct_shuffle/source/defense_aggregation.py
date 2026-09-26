"""Server-only aggregation baselines, with no access to attack labels."""

import numpy as np
from flwr.server.strategy.aggregate import aggregate

METHODS = ("fedavg", "clip_fedavg", "coordinate_median")
DEFINITIONS = {
    "fedavg": "Flower sample-count-weighted mean of submitted parameters",
    "clip_fedavg": (
        "delta_i = submitted_i - global; full-model float64 L2 norm; "
        "tau = unweighted median of submitted norms each round; "
        "factor_i = min(1, tau/norm_i), with factor=1 for zero norm; "
        "Flower sample-weighted mean of global + factor_i * delta_i"
    ),
    "coordinate_median": (
        "global + unweighted coordinate-wise median of submitted deltas; "
        "each logical identity is one vote; no clipping or sample weighting"
    ),
}


def l2_norm(arrays):
    return float(np.sqrt(sum(np.sum(np.square(a)) for a in arrays)))


def aggregate_updates(global_parameters, submissions, method):
    """Return native-dtype parameters and per-submission audit information.

    Norms and defenses use float64 deltas across every model tensor, including
    biases. Cast only when reconstructing submitted/native model parameters.
    FedAvg itself follows the existing Flower implementation exactly.
    """
    if method not in METHODS or not submissions or not global_parameters:
        raise ValueError("Unknown method or empty parameter/submission list")
    for reference in global_parameters:
        if not np.issubdtype(reference.dtype, np.floating):
            raise ValueError("Model parameters must be floating-point arrays")
        if not np.isfinite(reference).all():
            raise ValueError("Nonfinite global parameters")
    updates = []
    for parameters, count in submissions:
        if not np.isfinite(count) or count <= 0:
            raise ValueError("Reported sample counts must be positive")
        if len(parameters) != len(global_parameters):
            raise ValueError("Parameter tensor count mismatch")
        delta = []
        for value, reference in zip(parameters, global_parameters):
            if (
                value.shape != reference.shape
                or value.dtype != reference.dtype
            ):
                raise ValueError("Parameter shape/dtype mismatch")
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite submission")
            delta.append(
                value.astype(np.float64) - reference.astype(np.float64)
            )
        updates.append(delta)
    norms = np.array([l2_norm(delta) for delta in updates])
    if not np.isfinite(norms).all():
        raise ValueError("Update norm overflow")
    threshold = float(np.median(norms)) if method == "clip_fedavg" else None
    records = []
    reconstructed = []
    for delta, norm, (_, count) in zip(updates, norms, submissions):
        factor = 1.0
        if threshold is not None and norm > 0:
            factor = min(1.0, threshold / norm)
        clipped = [factor * value for value in delta]
        clipped_norm = l2_norm(clipped)
        if threshold is not None and clipped_norm > threshold + 1e-12 * max(
            1, threshold
        ):
            raise ValueError("Clipping bound violated")
        parameters = [
            (reference.astype(np.float64) + value).astype(reference.dtype)
            for reference, value in zip(global_parameters, clipped)
        ]
        reconstructed.append((parameters, count))
        records.append(
            {
                "submitted_update_l2": float(norm),
                "threshold": threshold,
                "clip_factor": factor,
                "clipped_update_l2": clipped_norm,
                "reconstructed_update_l2": l2_norm(
                    [
                        value.astype(np.float64) - reference.astype(np.float64)
                        for value, reference in zip(
                            parameters, global_parameters
                        )
                    ]
                ),
            }
        )
    if method == "fedavg":
        # aggregate may mutate its input arrays; snapshots remain independent.
        result = aggregate(
            [
                ([value.copy() for value in parameters], count)
                for parameters, count in submissions
            ]
        )
    elif method == "clip_fedavg":
        result = aggregate(reconstructed)
    else:
        result = [
            (
                reference.astype(np.float64)
                + np.median(
                    np.stack([delta[index] for delta in updates]), axis=0
                )
            ).astype(reference.dtype)
            for index, reference in enumerate(global_parameters)
        ]
    for value, reference in zip(result, global_parameters):
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("Aggregated shape/dtype changed")
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite aggregated parameters")
    return result, records
