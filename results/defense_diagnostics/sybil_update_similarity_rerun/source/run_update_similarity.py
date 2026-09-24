"""Scalar Sybil diagnostics; read-only preview unless --run is supplied."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_sybil_backdoor as sybil

pilot = sybil.pilot
sweep = sybil.sweep
OUTPUT = Path("results/defense_diagnostics/sybil_update_similarity_rerun")
DEFINITION = {
    "delta": "float64(submitted parameter) - float64(current global parameter)",
    "stage": "after local attack transformation, before FedAvg",
    "norm": "L2 over all flattened parameter tensors",
    "cosine": "dot(delta_i, delta_j) / (norm_i * norm_j)",
    "zero_norm": "undefined cosine stored as an empty CSV field",
    "pair_labels": "partition-0 replicas are Sybils in BOTH conditions",
    "threshold": None,
    "save_vectors": False,
}


def update_statistics(global_parameters, submissions):
    """Read submissions without mutation or random-number consumption."""
    deltas = []
    for parameters, _ in submissions:
        sweep.require(
            len(parameters) == len(global_parameters),
            "Parameter count differs",
        )
        for local, global_value in zip(parameters, global_parameters):
            sweep.require(
                local.shape == global_value.shape, "Parameter shape differs"
            )
        delta = np.concatenate(
            [
                (
                    local.astype(np.float64) - global_value.astype(np.float64)
                ).ravel()
                for local, global_value in zip(parameters, global_parameters)
            ]
        )
        sweep.require(np.isfinite(delta).all(), "Nonfinite submitted update")
        deltas.append(delta)
    norms = [float(np.linalg.norm(delta)) for delta in deltas]
    sweep.require(np.isfinite(norms).all(), "Nonfinite update norm")
    pairs = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            cosine = None
            if norms[i] != 0 and norms[j] != 0:
                cosine = float(
                    np.dot(deltas[i] / norms[i], deltas[j] / norms[j])
                )
                cosine = float(np.clip(cosine, -1.0, 1.0))
            pairs.append((i, j, cosine))
    return norms, pairs


def pair_type(i, j):
    """Post-computation labels never enter aggregation or diagnostics math."""
    replicas = {0, 5, 6}
    count = int(i in replicas) + int(j in replicas)
    return ("honest_honest", "sybil_honest", "sybil_sybil")[count]


def train_with_logging(workers, loader, folder, seed, mode):
    """Observe the unchanged pilot loop through a scoped aggregation wrapper."""
    current = [value.copy() for value in workers[0].get_parameters({})]
    original_aggregate = pilot.aggregate
    round_number = 0
    with (
        (folder / "update_norms.csv").open("x", newline="") as norm_file,
        (folder / "pairwise_similarity.csv").open(
            "x", newline=""
        ) as pair_file,
    ):
        norm_writer = csv.writer(norm_file, lineterminator="\n")
        pair_writer = csv.writer(pair_file, lineterminator="\n")
        norm_writer.writerow(["round", "client", "update_l2_norm"])
        pair_writer.writerow(
            ["round", "client_i", "client_j", "pair_type", "cosine_similarity"]
        )

        def observed_aggregate(submissions):
            nonlocal current, round_number
            norms, pairs = update_statistics(current, submissions)
            round_number += 1
            for identity, norm in enumerate(norms):
                norm_writer.writerow([round_number, identity, norm])
            for i, j, cosine in pairs:
                pair_writer.writerow(
                    [round_number, i, j, pair_type(i, j), cosine]
                )
            norm_file.flush()
            pair_file.flush()
            # Pass the original submissions to the original FedAvg unchanged.
            result = original_aggregate(submissions)
            current = [value.copy() for value in result]
            return result

        with patch.object(pilot, "aggregate", observed_aggregate):
            final = pilot.train_condition(workers, loader, folder, seed, mode)
    sweep.require(round_number == pilot.SETTINGS.rounds, "Missing rounds")
    return final


def preflight(pair, hashes):
    expected, configs, _ = sybil.preflight(pair, hashes, sweep.PILOT)
    sweep.require(
        sweep.inventory(sybil.OUTPUT)
        == sweep.read_json(sybil.OUTPUT / "checksums.json"),
        "Original Sybil archive is incomplete or changed",
    )
    for name in (*sweep.SOURCES, *sybil.EXTRA_SOURCES):
        sweep.require(
            sweep.sha256(Path(name))
            == sweep.sha256(sybil.OUTPUT / "source" / name),
            f"Source differs from validated Sybil archive: {name}",
        )
    sweep.require(
        sweep.read_json(sybil.OUTPUT / "dataset_hashes.json") == hashes,
        "Sybil dataset hashes differ",
    )
    sybil.validate_new_result(sybil.OUTPUT / "sybil_3", 3, expected, configs)
    return expected, configs


def save_provenance(output, hashes, configs):
    source = output / "source"
    source.mkdir()
    names = (
        *sweep.SOURCES,
        *sybil.EXTRA_SOURCES,
        "run_update_similarity.py",
        "tests/test_update_similarity.py",
    )
    for name in names:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    environment = sweep.read_json(sybil.OUTPUT / "environment.json")
    environment.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "source_sha256": sweep.inventory(source),
        }
    )
    pilot.write_json(output / "environment.json", environment)
    pilot.write_json(output / "dataset_hashes.json", hashes)
    pilot.write_json(
        output / "config.json",
        {
            "conditions": [
                configs[3, seed, attack]
                for seed in pilot.SETTINGS.seeds
                for attack in (False, True)
            ],
            "aggregation": "unchanged sample-weighted FedAvg",
            "diagnostics": DEFINITION,
            "new_runs": 6,
            "total_rounds": 60,
        },
    )
    pilot.write_json(
        output / "provenance.json",
        {
            "source_archive": str(sybil.OUTPUT.resolve()),
            "source_manifest_sha256": sweep.sha256(
                sybil.OUTPUT / "checksums.json"
            ),
            "source_compatibility": "exact bytes; original modules unchanged",
            "reused_runs": 0,
            "new_runs": 6,
            "diagnostics": DEFINITION,
        },
    )


def run(output, pair, hashes, expected, configs):
    output.mkdir(parents=True, exist_ok=False)
    save_provenance(output, hashes, configs)
    loader = DataLoader(
        pilot.asr_dataset(pair[1]), batch_size=32, shuffle=False
    )
    final = []
    for seed, (partitions, _, _) in expected.items():
        for attack in (False, True):
            mode = "backdoor" if attack else "clean"
            workers, parts, selected = sybil.make_workers(
                pair, partitions, seed, attack, 3
            )
            sybil.verify_workers(workers, partitions, selected, attack, 3)
            config = sybil.condition_config(
                workers, partitions, selected, seed, attack, 3
            )
            sweep.require(config == configs[3, seed, attack], "Config drift")
            folder = output / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            pilot.write_json(folder / "config.json", config)
            np.savez_compressed(
                folder / "partitions.npz",
                **{
                    f"client_{i}": np.asarray(part, dtype=np.int64)
                    for i, part in enumerate(parts)
                },
            )
            final.append(
                train_with_logging(workers, loader, folder, seed, mode)
            )
            sweep.require(
                pilot.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
                "Dataset changed during training",
            )
            # Exact per-round comparison to the validated original trajectory.
            original = sybil.OUTPUT / "sybil_3" / f"seed_{seed}" / mode
            sweep.require(
                (folder / "results.csv").read_bytes()
                == (original / "results.csv").read_bytes(),
                "Logged run differs from original metrics; stop for audit",
            )
    sweep.write_csv(output / "per_seed.csv", sybil.paired_rows(3, final))
    pilot.write_json(output / "summary.json", pilot.summarize(final))
    pilot.write_json(output / "checksums.json", sweep.inventory(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        sweep.require_fresh_output(args.output)
        protected = (
            sybil.OUTPUT,
            sweep.PILOT,
            Path("results/defense_baselines"),
            Path("results/defense_diagnostics/update_similarity"),
        )
        sweep.require(
            not any(
                args.output.resolve().is_relative_to(p.resolve())
                for p in protected
            ),
            "Protected result directory",
        )
        sweep.require(not sys.flags.optimize, "Run without -O")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        pair, hashes = sweep.current_data()
        expected, configs = preflight(pair, hashes)
        print("6 new runs / 60 rounds; 7 norms and 21 pairs per round.")
        print("Zero-norm cosine: blank; no threshold; no vectors saved.")
        if not args.run:
            print("Preview passed: no training, downloads, or result writes.")
            return
        run(args.output, pair, hashes, expected, configs)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.exit(1, f"Diagnostic stopped: {error}\n")


if __name__ == "__main__":
    main()
