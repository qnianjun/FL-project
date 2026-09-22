"""Validated Backdoor alpha sweep; preview by default, train only with --run."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

import run_backdoor as pilot


ALPHAS = (0.01, 0.1, 1, 10, 100)
PILOT = Path("results/Backdoor_test/mnist_alpha_0.1_ratio_0.2")
OUTPUT = Path("results/Backdoor_test/alpha_sweep")
SOURCES = (
    "run_backdoor.py", "client.py", "run_balanced.py",
    "balanced_partition.py", "config.json", "tests/test_backdoor.py",
)
METRICS = ("clean_loss", "clean_accuracy_percent", "asr_percent")
BASE_SETTINGS = {
    "alpha": 0.1, "poison_ratio": 0.2, "target_label": 0,
    "trigger_size": 3, "trigger_row": 25, "trigger_column": 25,
    "trigger_value": 1.0, "seeds": [42, 43, 44], "num_clients": 5,
    "attacker": 0, "rounds": 10, "local_epochs": 1, "batch_size": 32,
    "learning_rate": 0.01, "optimizer": "SGD",
}
PROTOCOL = {
    "partition_method": "capacity_constrained_dirichlet_v1",
    "aggregation": "sample-weighted FedAvg; 20% per client",
    "poison_selection": "floor(ratio * attacker count); fixed/seed",
    "poison_selection_seed": "seed + 2000",
    "shuffle_seed": "seed + 1000 + client_id",
    "asr_denominator": "test images: original label != target",
    "execution": "sequential CPU; all clients; no update scaling",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    paths = sorted(folder.rglob("*"))
    require(not any(path.is_symlink() for path in paths),
            f"Symlinks are not accepted: {folder}")
    return {
        str(path.relative_to(folder)): sha256(path)
        for path in paths
        if path.is_file() and path != folder / "checksums.json"
    }


def settings_for(alpha):
    return {**BASE_SETTINGS, "alpha": alpha}


def expected_files():
    names = {
        "config.json", "dataset_hashes.json", "environment.json",
        "summary.json",
    }
    names.update(f"source/{name}" for name in SOURCES)
    for seed in BASE_SETTINGS["seeds"]:
        for mode in ("clean", "backdoor"):
            names.update(
                f"seed_{seed}/{mode}/{name}"
                for name in ("config.json", "partitions.npz", "results.csv")
            )
    return names


def verify_checksums(folder):
    require(folder.is_dir() and not folder.is_symlink(),
            f"Missing or invalid result directory: {folder}")
    saved = read_json(folder / "checksums.json")
    require(set(saved) == expected_files(),
            f"Incomplete or unexpected artifact inventory: {folder}")
    require(inventory(folder) == saved, f"Checksum mismatch: {folder}")


def current_data():
    # Preview never downloads or changes the MNIST cache.
    pair = tuple(
        datasets.MNIST(
            "./data", train=train, download=False,
            transform=transforms.ToTensor(),
        )
        for train in (True, False)
    )
    require(tuple(map(len, pair)) == (60000, 10000), "Unexpected MNIST size")
    hashes = {
        "loaded_arrays_sha256": pilot.dataset_hashes(pair),
        "raw_files_sha256": {
            str(path): sha256(path)
            for path in sorted(Path("data/MNIST/raw").iterdir())
            if path.is_file() and path.suffix != ".gz"
        },
    }
    return pair, hashes


def expected_pairs(pair, alpha):
    """Regenerate partitions and both initial identities, without training."""
    result = {}
    for seed in BASE_SETTINGS["seeds"]:
        partitions = pilot.balanced_partition(pair[0].targets, 5, alpha, seed)
        require([len(part) for part in partitions] == [12000] * 5,
                f"Unequal client counts: alpha={alpha}, seed={seed}")
        require(np.array_equal(np.sort(np.concatenate(partitions)),
                               np.arange(60000)), "Partition coverage mismatch")
        clean, selected = pilot.make_workers(pair, partitions, seed, False)
        attack, repeated = pilot.make_workers(pair, partitions, seed, True)
        identity = pilot.identity(clean, partitions)
        require(identity == pilot.identity(attack, partitions),
                "Clean/backdoor initial identities differ")
        require(np.array_equal(selected, repeated) and len(selected) == 2400,
                "Poison selection mismatch")
        result[seed] = (partitions, identity, selected)
    return result


def read_metrics(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["round", *METRICS],
                f"Unexpected metrics columns: {path}")
        rows = list(reader)
    require(len(rows) == 10, f"Incomplete metrics: {path}")
    parsed = []
    for number, row in enumerate(rows, 1):
        require(set(row) == {"round", *METRICS} and row["round"] == str(number),
                f"Invalid round sequence: {path}")
        values = {metric: float(row[metric]) for metric in METRICS}
        require(all(math.isfinite(value) for value in values.values()),
                f"Nonfinite metrics: {path}")
        require(values["clean_loss"] >= 0 and all(
            0 <= values[metric] <= 100 for metric in METRICS[1:]
        ), f"Out-of-range metrics: {path}")
        parsed.append({"round": number, **values})
    return parsed[-1]


def validate_result(folder, alpha, pair, hashes, expected):
    verify_checksums(folder)
    require(read_json(folder / "config.json") == {
        **settings_for(alpha), **PROTOCOL,
    }, f"Incompatible configuration: {folder}")
    require(read_json(folder / "dataset_hashes.json") == hashes,
            f"Dataset hashes differ: {folder}")
    environment = read_json(folder / "environment.json")
    source_hashes = {name: sha256(Path(name)) for name in SOURCES}
    require(environment["source_sha256"] == source_hashes,
            f"Current source differs from saved source: {folder}")
    for name, digest in source_hashes.items():
        require(sha256(folder / "source" / name) == digest,
                f"Saved source hash mismatch: {name}")
    require(environment["python"] == platform.python_version(),
            "Python version differs from pilot")
    require(environment["packages"] == {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy", "flwr")
    }, "Package versions differ from pilot")
    require(environment["torch_threads"] == 1
            and environment["deterministic_algorithms"] is True,
            "Execution settings differ from pilot")
    final = []
    for seed, (partitions, identity, selected) in expected.items():
        for mode in ("clean", "backdoor"):
            local = folder / f"seed_{seed}" / mode
            expected_config = {
                **settings_for(alpha), **identity,
                "seed": seed, "condition": mode,
                "poison_selection_seed": seed + 2000,
                "poisoned_local_positions": (
                    selected.tolist() if mode == "backdoor" else []
                ),
                "client_label_counts": [
                    np.bincount(np.asarray(pair[0].targets)[part],
                                minlength=10).tolist()
                    for part in partitions
                ],
                "asr_sample_count": len(pilot.asr_dataset(pair[1])),
                "validation": "identity, transforms, data hashes passed",
            }
            require(read_json(local / "config.json") == expected_config,
                    f"Incompatible pair configuration: {local}")
            with np.load(local / "partitions.npz", allow_pickle=False) as saved:
                require(set(saved.files) == {f"client_{i}" for i in range(5)},
                        f"Unexpected partition keys: {local}")
                for cid, partition in enumerate(partitions):
                    actual = saved[f"client_{cid}"]
                    require(actual.dtype == np.dtype("int64")
                            and np.array_equal(actual, partition),
                            f"Partition mismatch: {local}, client={cid}")
            final.append({"seed": seed, "condition": mode,
                          **read_metrics(local / "results.csv")})
    require(read_json(folder / "summary.json") == pilot.summarize(final),
            f"Summary differs from raw metrics: {folder}")
    return final


def paired_rows(alpha, final):
    rows = []
    for seed in BASE_SETTINGS["seeds"]:
        pair = [row for row in final if row["seed"] == seed]
        require(len(pair) == 2 and {row["condition"] for row in pair}
                == {"clean", "backdoor"}, f"Missing or duplicate pair: {seed}")
        by_mode = {row["condition"]: row for row in pair}
        clean = by_mode["clean"]
        attack = by_mode["backdoor"]
        rows.append({
            "alpha": alpha, "seed": seed, "round": 10,
            "clean_accuracy_percent": clean["clean_accuracy_percent"],
            "backdoor_clean_accuracy_percent": attack["clean_accuracy_percent"],
            "clean_asr_percent": clean["asr_percent"],
            "backdoor_asr_percent": attack["asr_percent"],
            "paired_clean_accuracy_change_pp": (
                attack["clean_accuracy_percent"] - clean["clean_accuracy_percent"]
            ),
            "paired_asr_increase_pp": attack["asr_percent"] - clean["asr_percent"],
        })
    return rows


def summary_rows(rows):
    result = []
    for alpha in ALPHAS:
        subset = [row for row in rows if row["alpha"] == alpha]
        require(len(subset) == 3 and {row["seed"] for row in subset}
                == set(BASE_SETTINGS["seeds"]), f"Incomplete alpha={alpha}")
        summary = {"alpha": alpha, "n_seeds": 3, "round": 10}
        for metric in subset[0]:
            if metric in ("alpha", "seed", "round"):
                continue
            values = [row[metric] for row in subset]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sample_sd"] = statistics.stdev(values)
        result.append(summary)
    return result


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_fresh_output(output):
    require(not output.exists() and not output.is_symlink(),
            f"Output already exists; refusing to overwrite or resume: {output}")


def train_alpha(alpha, destination):
    """Only replace alpha; trigger defaults retain all original pilot settings."""
    original_settings = pilot.SETTINGS
    original_argv = sys.argv
    try:
        pilot.SETTINGS = replace(original_settings, alpha=alpha)
        sys.argv = ["run_backdoor.py", "--run", "--output", str(destination)]
        pilot.main()
    finally:
        pilot.SETTINGS = original_settings
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start full training")
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        require_fresh_output(args.output)
        require(not args.output.resolve().is_relative_to(args.pilot.resolve()),
                "Output must not be inside the existing pilot")
        require(json.loads(json.dumps(asdict(pilot.SETTINGS))) == BASE_SETTINGS,
                "Pilot settings changed")
        require(not sys.flags.optimize, "Run without -O: pilot checks use assert")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        pair, hashes = current_data()
        expected = {alpha: expected_pairs(pair, alpha) for alpha in ALPHAS}
        final_pilot = validate_result(
            args.pilot, 0.1, pair, hashes, expected[0.1]
        )
        print(f"Validated pilot reuse: {args.pilot}", flush=True)
        for alpha in ALPHAS:
            action = "reuse 6 completed runs" if alpha == 0.1 else "train 6 runs"
            print(f"alpha={alpha}: {action}; partitions/matched identities passed")
        print("24 new runs, 240 rounds; final-round paired differences in pp.")
        if not args.run:
            print("Preview passed. No training started and no results written.")
            return
        args.output.mkdir(parents=True, exist_ok=False)
        source = args.output / "source"
        source.mkdir()
        for name in ("run_backdoor_sweep.py", "tests/test_backdoor_sweep.py"):
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, destination)
        pilot.write_json(args.output / "provenance.json", {
            "alphas": ALPHAS, "settings": BASE_SETTINGS,
            "pilot_path": str(args.pilot.resolve()),
            "pilot_checksums_sha256": sha256(args.pilot / "checksums.json"),
            "reuse_policy": "exact source, environment, data, partitions, metrics",
            "paired_difference": "backdoor minus clean; percentage points",
            "source_sha256": inventory(source),
        })
        rows = []
        for alpha in ALPHAS:
            destination = args.output / f"alpha_{alpha}"
            if alpha == 0.1:
                # Preserve every original byte, including the original provenance.
                shutil.copytree(args.pilot, destination)
                final = validate_result(destination, alpha, pair, hashes,
                                        expected[alpha])
                require(final == final_pilot, "Pilot changed after preflight")
            else:
                train_alpha(alpha, destination)
                final = validate_result(destination, alpha, pair, hashes,
                                        expected[alpha])
            rows.extend(paired_rows(alpha, final))
        write_csv(args.output / "per_seed.csv", rows)
        write_csv(args.output / "summary.csv", summary_rows(rows))
        pilot.write_json(args.output / "checksums.json", inventory(args.output))
        print(f"Sweep complete: {args.output}")
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.exit(1, f"Sweep stopped: {error}\n")


if __name__ == "__main__":
    main()
