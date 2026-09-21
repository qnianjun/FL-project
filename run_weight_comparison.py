"""Run Experiment D with the Alpha=0.1 partition used by Experiment A.

Experiment D keeps the images and training seeds fixed, then compares the
normal sample-weighted FedAvg rule with an equal-weight client average.  The
default command runs only the new equal-weight condition; the sample-weighted
results already exist in Experiment A.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flwr.server.strategy.aggregate import aggregate

import client


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    ROOT
    / "results"
    / "experiments"
    / "2026-09-20_alpha_0.1_scale_10_three_seeds"
)
SEEDS = [42, 43, 44]
NUM_CLIENTS = 5
ATTACK_SCALE = 10


class FixedPartitionClient(client.FlowerClient):
    """Use a saved partition while reusing the tested training and attack code."""

    def __init__(self, cid, datasets_pair, partition, seed):
        self.cid = int(cid)
        torch.manual_seed(seed)
        self.model = client.Net()
        self.trainset = Subset(datasets_pair[0], partition)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 1000 + self.cid),
        )
        self.testset = datasets_pair[1]
        self.testloader = DataLoader(self.testset, batch_size=32, shuffle=False)


def write_json(path, value):
    """Write readable JSON and always terminate the file with a newline."""

    path.write_text(json.dumps(value, indent=2) + "\n")


def array_digest(values):
    """Return a stable digest for a list of NumPy arrays."""

    content = b"".join(np.asarray(value).tobytes() for value in values)
    return hashlib.sha256(content).hexdigest()


def load_saved_partitions(source):
    """Load one original A partition for each seed and validate its coverage."""

    partitions_by_seed = {}
    for seed in SEEDS:
        partition_file = source / f"seed_{seed}" / "clean" / "partitions.npz"
        if not partition_file.exists():
            raise FileNotFoundError(f"Could not find saved partition: {partition_file}")

        with np.load(partition_file) as archive:
            partitions = [
                archive[f"client_{cid}"].astype(np.int64)
                for cid in range(NUM_CLIENTS)
            ]

        all_indices = np.concatenate(partitions)
        if not np.array_equal(np.sort(all_indices), np.arange(60000)):
            raise ValueError(
                f"Seed {seed} partition must contain every MNIST training sample once"
            )
        partitions_by_seed[seed] = partitions

    return partitions_by_seed


def equal_average(results):
    """Average each model layer equally across the five clients."""

    layers = zip(*(weights for weights, _ in results), strict=True)
    return [np.mean(layer_values, axis=0) for layer_values in layers]


def combine(results, aggregation_mode):
    """Apply the requested aggregation rule to client model updates."""

    if aggregation_mode == "equal":
        return equal_average(results)
    if aggregation_mode == "weighted":
        return aggregate(results)
    raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")


def save_environment(output, source):
    """Save source snapshots and versions needed to audit a new run."""

    source_copy = output / "source"
    source_copy.mkdir()
    for name in ["client.py", "run_weight_comparison.py", "config.json"]:
        shutil.copy2(ROOT / name, source_copy / name)

    write_json(
        output / "environment.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_partition": str(source),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "numpy", "flwr"]
            },
            "torch_threads": 1,
            "deterministic_algorithms": True,
        },
    )


def run_condition(output, datasets_pair, partitions_by_seed, aggregation_mode):
    """Run three seeds, each with a clean and a poisoned paired condition."""

    final_rows = []
    for seed in SEEDS:
        partitions = partitions_by_seed[seed]
        for attack_enabled in [False, True]:
            mode = "poisoned" if attack_enabled else "clean"
            folder = output / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            client.SEED = seed
            client.ENABLE_POISON = attack_enabled
            client.POISON_CLIENT = 0
            client.POISON_SCALE = ATTACK_SCALE
            workers = [
                FixedPartitionClient(cid, datasets_pair, partitions[cid], seed)
                for cid in range(NUM_CLIENTS)
            ]
            initial = workers[0].get_parameters({})
            identity = {
                "initial_parameters_sha256": array_digest(initial),
                "partition_sha256": [array_digest([part]) for part in partitions],
            }
            write_json(
                folder / "config.json",
                {
                    "seed": seed,
                    "alpha": 0.1,
                    "aggregation": aggregation_mode,
                    "enable_poison": attack_enabled,
                    "poison_client": 0,
                    "poison_scale": ATTACK_SCALE,
                    "rounds": 10,
                    "num_clients": NUM_CLIENTS,
                    "batch_size": 32,
                    "local_epochs": 1,
                    "optimizer": "SGD",
                    "learning_rate": 0.01,
                    "client_sample_counts": [len(part) for part in partitions],
                    "shuffle_seeds": [seed + 1000 + cid for cid in range(NUM_CLIENTS)],
                    **identity,
                },
            )
            parameters = initial
            with (folder / "results.csv").open("w", newline="") as result_file, (
                folder / "updates.csv"
            ).open("w", newline="") as update_file:
                result_writer = csv.writer(result_file, lineterminator="\n")
                update_writer = csv.writer(update_file, lineterminator="\n")
                result_writer.writerow(["round", "loss", "accuracy"])
                update_writer.writerow(
                    ["round", "client", "update_norm", "sent_update_norm", "scale_ratio"]
                )
                for round_number in range(1, 11):
                    updates = []
                    for worker in workers:
                        weights, count, metrics = worker.fit(parameters, {})
                        norm = metrics["update_norm"]
                        sent_norm = metrics["sent_update_norm"]
                        expected_scale = ATTACK_SCALE if attack_enabled and worker.cid == 0 else 1
                        if not math.isclose(
                            sent_norm,
                            expected_scale * norm,
                            rel_tol=1e-5,
                            abs_tol=1e-7,
                        ):
                            raise AssertionError("The submitted update has an unexpected scale")
                        update_writer.writerow(
                            [round_number, worker.cid, norm, sent_norm, sent_norm / norm]
                        )
                        updates.append((weights, count))
                    parameters = combine(updates, aggregation_mode)
                    loss, _, metrics = workers[0].evaluate(parameters, {})
                    accuracy = metrics["accuracy"]
                    result_writer.writerow([round_number, loss, accuracy])
                    print(
                        f"{aggregation_mode} seed={seed} {mode} "
                        f"round={round_number}/10 accuracy={accuracy:.4f}",
                        flush=True,
                    )
            final_rows.append(
                {
                    "seed": seed,
                    "condition": mode,
                    "accuracy": accuracy,
                    "loss": loss,
                }
            )
    return final_rows


def save_summary(output, rows, partitions_by_seed):
    """Save final-round values and paired damage for the new aggregation run."""

    clean = {row["seed"]: row for row in rows if row["condition"] == "clean"}
    poisoned = {row["seed"]: row for row in rows if row["condition"] == "poisoned"}
    paired = []
    for seed in SEEDS:
        clean_row = clean[seed]
        poisoned_row = poisoned[seed]
        paired.append(
            {
                "seed": seed,
                "clean_accuracy": clean_row["accuracy"],
                "poisoned_accuracy": poisoned_row["accuracy"],
                "clean_loss": clean_row["loss"],
                "poisoned_loss": poisoned_row["loss"],
                "accuracy_damage_pp": 100 * (clean_row["accuracy"] - poisoned_row["accuracy"]),
                "loss_damage": poisoned_row["loss"] - clean_row["loss"],
            }
        )
    with (output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    summary = {
        "aggregation": json.loads((output / "seed_42" / "clean" / "config.json").read_text())["aggregation"],
        "alpha": 0.1,
        "scale": ATTACK_SCALE,
        "samples_per_client_by_seed": {
            str(seed): [len(part) for part in partitions]
            for seed, partitions in partitions_by_seed.items()
        },
        "seeds": SEEDS,
    }
    for metric in [
        "clean_accuracy",
        "poisoned_accuracy",
        "clean_loss",
        "poisoned_loss",
        "accuracy_damage_pp",
        "loss_damage",
    ]:
        values = [row[metric] for row in paired]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_sample_std"] = statistics.stdev(values)
    write_json(output / "summary.json", summary)
    write_json(
        output / "checksums.json",
        {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )


def save_weight_comparison(output, weighted_source, equal_rows):
    """Compare the new equal-weight rows with the saved weighted rows."""

    weighted_summary = json.loads((weighted_source / "summary.json").read_text())
    weighted_rows = {
        (row["seed"], row["condition"]): row for row in weighted_summary["runs"]
    }
    equal_by_key = {
        (row["seed"], row["condition"]): row for row in equal_rows
    }
    comparison = []
    for seed in SEEDS:
        clean_weighted = weighted_rows[(seed, "clean")]
        poison_weighted = weighted_rows[(seed, "poisoned")]
        clean_equal = equal_by_key[(seed, "clean")]
        poison_equal = equal_by_key[(seed, "poisoned")]
        weighted_damage = 100 * (
            clean_weighted["accuracy"] - poison_weighted["accuracy"]
        )
        equal_damage = 100 * (
            clean_equal["accuracy"] - poison_equal["accuracy"]
        )
        comparison.append(
            {
                "seed": seed,
                "weighted_clean_accuracy": clean_weighted["accuracy"],
                "weighted_poisoned_accuracy": poison_weighted["accuracy"],
                "weighted_damage_pp": weighted_damage,
                "equal_clean_accuracy": clean_equal["accuracy"],
                "equal_poisoned_accuracy": poison_equal["accuracy"],
                "equal_damage_pp": equal_damage,
                "equal_minus_weighted_damage_pp": equal_damage - weighted_damage,
            }
        )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(comparison[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparison)

    # comparison.csv is created after the first checksum pass in save_summary.
    # Refresh it so every generated output is covered by the audit file.
    write_json(
        output / "checksums.json",
        {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", choices=["equal", "weighted"], default="equal")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "studies" / "weight_comparison" / "equal",
    )
    parser.add_argument("--run", action="store_true", help="Start the six training runs")
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f"Source experiment does not exist: {args.source}")
    partitions_by_seed = load_saved_partitions(args.source)
    print(f"Alpha=0.1, scale=10, aggregation={args.aggregation}")
    print(
        "Client sample counts by seed: "
        f"{ {seed: [len(part) for part in parts] for seed, parts in partitions_by_seed.items()} }"
    )
    print(f"Output: {args.output}")
    if not args.run:
        print("Plan only. Add --run to start six runs: 3 seeds x Clean/Poisoned.")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    save_environment(args.output, args.source)
    datasets_pair = client.load_data()
    write_json(
        args.output / "dataset_hashes.json",
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "data" / "MNIST" / "raw").iterdir())
            if path.is_file() and path.suffix != ".gz"
        },
    )
    run_condition(args.output, datasets_pair, partitions_by_seed, args.aggregation)
    # The rows are computed once so the six runs are never repeated accidentally.
    rows = []
    for seed in SEEDS:
        for mode in ["clean", "poisoned"]:
            result_rows = list(csv.DictReader((args.output / f"seed_{seed}" / mode / "results.csv").open()))
            final = result_rows[-1]
            rows.append({"seed": seed, "condition": mode, "accuracy": float(final["accuracy"]), "loss": float(final["loss"])})
    save_summary(args.output, rows, partitions_by_seed)
    save_weight_comparison(args.output, args.source, rows)
    print(f"Complete: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
