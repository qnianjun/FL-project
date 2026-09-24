"""Matched defense baselines; preview only unless explicitly given --run."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import client
from defense_aggregation import DEFINITIONS, METHODS, aggregate_updates
import run_backdoor as backdoor
import run_backdoor_sweep as sweep
import run_experiment_c as experiment_c
import run_sybil_backdoor as sybil
from source_compatibility import same_python_structure

BENCHMARKS = ("model_poisoning", "backdoor", "sybil_backdoor")
SEEDS = (42, 43, 44)
MODEL_SOURCE = Path("results/studies/equal_samples/alpha_0.1_scale_10")
SYBIL_SOURCE = sybil.OUTPUT
OUTPUT = Path("results/defense_baselines/pilot_alpha_0.1")
SOURCES = tuple(
    dict.fromkeys(
        (
            *sweep.SOURCES,
            *sybil.EXTRA_SOURCES,
            "run_experiment_c.py",
            "source_compatibility.py",
            "defense_aggregation.py",
            "run_defense_pilot.py",
            "tests/test_defense_pilot.py",
        )
    )
)


def verify_archive(folder):
    manifest = sweep.read_json(folder / "checksums.json")
    sweep.require(
        sweep.inventory(folder) == manifest,
        f"Incomplete, changed or extra artifacts: {folder}",
    )
    return manifest


def verify_environment(folder, structural_sources=()):
    environment = sweep.read_json(folder / "environment.json")
    sweep.require(
        environment["python"] == platform.python_version(),
        "Python version differs from baseline",
    )
    sweep.require(
        environment["packages"]
        == {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "flwr")
        },
        "Package versions differ from baseline",
    )
    sweep.require(
        environment["torch_threads"] == 1
        and environment["deterministic_algorithms"] is True,
        "Execution settings differ from baseline",
    )
    compatibility = {}
    for name, saved_hash in environment["source_sha256"].items():
        saved = folder / "source" / name
        current = Path(name)
        sweep.require(
            sweep.sha256(saved) == saved_hash, "Saved source changed"
        )
        exact = sweep.sha256(current) == saved_hash
        structural = name in structural_sources and same_python_structure(
            saved, current
        )
        sweep.require(exact or structural, f"Incompatible source: {name}")
        compatibility[name] = {
            "saved_sha256": saved_hash,
            "current_sha256": sweep.sha256(current),
            "match": "bytes" if exact else "identical Python AST",
        }
    return compatibility


def make_workers(benchmark, pair, partitions, seed, attack):
    count = 3 if benchmark == "sybil_backdoor" else 1
    poison_data = attack and benchmark != "model_poisoning"
    workers, logical_parts, selected = sybil.make_workers(
        pair, partitions, seed, poison_data, count
    )
    sybil.verify_workers(workers, partitions, selected, poison_data, count)
    return workers, logical_parts, selected


def run_config(benchmark, method, workers, parts, selected, seed, attack):
    count = 3 if benchmark == "sybil_backdoor" else 1
    return {
        **sweep.BASE_SETTINGS,
        "benchmark": benchmark,
        "method": method,
        "defense_definition": DEFINITIONS[method],
        "seed": seed,
        "condition": "attack" if attack else "clean",
        "num_clients": len(workers),
        "original_num_partitions": 5,
        "logical_to_original_partition": sybil.partition_mapping(count),
        "sybil_count": count,
        "unique_attacker_sample_count": 12000,
        "fedavg_coalition_weight": sybil.coalition_weight(count),
        "poison_scale": 10 if benchmark == "model_poisoning" else 1,
        "poison_ratio": 0 if benchmark == "model_poisoning" else 0.2,
        "poisoned_local_positions": (
            selected.tolist()
            if attack and benchmark != "model_poisoning"
            else []
        ),
        "poison_selection_seed": seed + 2000,
        "asr_sample_count": None if benchmark == "model_poisoning" else 9020,
        **backdoor.identity(workers, parts),
    }


def validate_model_source(folder, pair, hashes, expected):
    names = {"environment.json", "dataset_hashes.json", "summary.json"}
    names.update(
        f"source/{name}"
        for name in (
            "client.py",
            "run_balanced.py",
            "balanced_partition.py",
            "config.json",
        )
    )
    for seed in SEEDS:
        for mode in ("clean", "poisoned"):
            names.update(
                f"seed_{seed}/{mode}/{name}"
                for name in (
                    "config.json",
                    "partitions.npz",
                    "results.csv",
                    "updates.csv",
                )
            )
    sweep.require(
        set(verify_archive(folder)) == names, "Incomplete model baseline"
    )
    compatibility = verify_environment(
        folder, ("client.py", "run_balanced.py", "balanced_partition.py")
    )
    sweep.require(
        sweep.read_json(folder / "dataset_hashes.json")
        == hashes["raw_files_sha256"],
        "Model dataset hashes differ",
    )
    experiment_c.validate_run(folder, 0.1)
    final = []
    for seed, (partitions, identity, _) in expected.items():
        for mode in ("clean", "poisoned"):
            local = folder / f"seed_{seed}" / mode
            cfg = sweep.read_json(local / "config.json")
            for key, value in identity.items():
                sweep.require(
                    cfg[key] == value, f"Model baseline differs: {key}"
                )
            counts = [
                np.bincount(
                    np.asarray(pair[0].targets)[p], minlength=10
                ).tolist()
                for p in partitions
            ]
            sweep.require(
                cfg["client_label_counts"] == counts, "Label counts differ"
            )
            with np.load(
                local / "partitions.npz", allow_pickle=False
            ) as saved:
                for cid, partition in enumerate(partitions):
                    sweep.require(
                        np.array_equal(saved[f"client_{cid}"], partition),
                        "Model partition differs",
                    )
            metric = experiment_c.read_rows(local / "results.csv")[-1]
            final.append(
                {
                    "seed": seed,
                    "condition": mode,
                    "accuracy": float(metric["accuracy"]),
                    "loss": float(metric["loss"]),
                }
            )
    clean = [row["accuracy"] for row in final if row["condition"] == "clean"]
    attack = [
        row["accuracy"] for row in final if row["condition"] == "poisoned"
    ]
    drops = [(a - b) * 100 for a, b in zip(clean, attack)]
    sweep.require(
        sweep.read_json(folder / "summary.json")
        == {
            "runs": final,
            "clean_mean_accuracy": statistics.mean(clean),
            "clean_sample_std": statistics.stdev(clean),
            "poisoned_mean_accuracy": statistics.mean(attack),
            "poisoned_sample_std": statistics.stdev(attack),
            "paired_drops_percentage_points": drops,
            "mean_drop_percentage_points": statistics.mean(drops),
            "drop_sample_std_percentage_points": statistics.stdev(drops),
        },
        "Model summary differs from raw metrics",
    )
    return compatibility


def read_final(benchmark, folder):
    if benchmark == "model_poisoning":
        row = experiment_c.read_rows(folder / "results.csv")[-1]
        return {
            "clean_loss": float(row["loss"]),
            "clean_accuracy_percent": float(row["accuracy"]) * 100,
            "asr_percent": None,
        }
    row = sweep.read_metrics(folder / "results.csv")
    return {name: row[name] for name in sweep.METRICS}


def preflight(pair, hashes, sources):
    expected, sybil_configs, _ = sybil.preflight(
        pair, hashes, sources["backdoor"]
    )
    compatibility = {
        "model_poisoning": validate_model_source(
            sources["model_poisoning"], pair, hashes, expected
        ),
        "backdoor": verify_environment(sources["backdoor"]),
    }
    sybil_root = sources["sybil_backdoor"]
    verify_archive(sybil_root)
    compatibility["sybil_backdoor"] = verify_environment(sybil_root)
    sweep.require(
        sweep.read_json(sybil_root / "dataset_hashes.json") == hashes,
        "Sybil dataset differs",
    )
    cfg = sweep.read_json(sybil_root / "config.json")
    sweep.require(
        cfg["original_partition_settings"] == sweep.BASE_SETTINGS
        and cfg["sybil_counts"] == [1, 2, 3],
        "Sybil settings differ",
    )
    provenance = sweep.read_json(sybil_root / "provenance.json")
    sweep.require(
        provenance["pilot_checksums_sha256"]
        == sweep.sha256(sources["backdoor"] / "checksums.json"),
        "Sybil pilot provenance differs",
    )
    sybil.validate_new_result(
        sybil_root / "sybil_3", 3, expected, sybil_configs
    )
    configs = {}
    baselines = {}
    for benchmark in BENCHMARKS:
        for seed, (partitions, _, _) in expected.items():
            pair_identity = None
            for attack in (False, True):
                workers, parts, selected = make_workers(
                    benchmark, pair, partitions, seed, attack
                )
                identity = backdoor.identity(workers, parts)
                if pair_identity is None:
                    pair_identity = identity
                sweep.require(
                    identity == pair_identity, "Defense pair differs"
                )
                mode = (
                    "poisoned"
                    if benchmark == "model_poisoning"
                    else "backdoor"
                )
                root = sources[benchmark]
                if benchmark == "sybil_backdoor":
                    root = root / "sybil_3"
                original = (
                    root / f"seed_{seed}" / (mode if attack else "clean")
                )
                saved = sweep.read_json(original / "config.json")
                for key, value in identity.items():
                    sweep.require(
                        saved[key] == value, f"Baseline mismatch: {key}"
                    )
                baselines[benchmark, seed, attack] = {
                    "path": original,
                    "metrics": read_final(benchmark, original),
                    "sha256": sweep.inventory(original),
                }
                for method in METHODS:
                    configs[benchmark, method, seed, attack] = run_config(
                        benchmark,
                        method,
                        workers,
                        parts,
                        selected,
                        seed,
                        attack,
                    )
    sweep.require(
        backdoor.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
        "Dataset mutated in preview",
    )
    print(
        "Validated all 18 FedAvg condition runs and 54 matched configurations."
    )
    return expected, configs, baselines, compatibility


def local_submissions(workers, parameters, benchmark, attack):
    submissions = []
    for worker in workers:
        if benchmark == "model_poisoning":
            # Use the validated fit implementation, including update scaling.
            with patch.multiple(
                client, ENABLE_POISON=attack, POISON_CLIENT=0, POISON_SCALE=10
            ):
                weights, count, _ = worker.fit(parameters, {})
        else:
            worker.set_parameters(parameters)
            client.train(worker.model, worker.trainloader)
            weights, count = worker.get_parameters({}), len(worker.trainset)
        submissions.append((weights, count))
    return submissions


def train_condition(workers, benchmark, method, attack, folder, asr_loader):
    parameters = workers[0].get_parameters({})
    metrics = []
    with (folder / "aggregation.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round",
                "logical_id",
                "submitted_update_l2",
                "threshold",
                "clip_factor",
                "clipped_update_l2",
                "reconstructed_update_l2",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for number in range(1, 11):
            submissions = local_submissions(
                workers, parameters, benchmark, attack
            )
            parameters, logs = aggregate_updates(
                parameters, submissions, method
            )
            for logical_id, log in enumerate(logs):
                writer.writerow(
                    {"round": number, "logical_id": logical_id, **log}
                )
            handle.flush()
            workers[0].set_parameters(parameters)
            loss, accuracy = client.test(
                workers[0].model, workers[0].testloader
            )
            asr = None
            if benchmark != "model_poisoning":
                _, asr = client.test(workers[0].model, asr_loader)
                asr *= 100
            row = {
                "round": number,
                "clean_loss": loss,
                "clean_accuracy_percent": accuracy * 100,
                "asr_percent": asr,
            }
            sweep.require(
                np.isfinite(loss)
                and loss >= 0
                and 0 <= accuracy <= 1
                and (asr is None or 0 <= asr <= 100),
                "Invalid metrics",
            )
            metrics.append(row)
            print(
                f"{benchmark} {method} attack={attack} round={number} "
                f"accuracy={accuracy:.4f} ASR={asr}",
                flush=True,
            )
    sweep.write_csv(folder / "results.csv", metrics)
    return {name: metrics[-1][name] for name in sweep.METRICS}


def paired_rows(final):
    rows = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            for seed in SEEDS:
                clean = final[benchmark, method, seed, False]
                attack = final[benchmark, method, seed, True]
                baseline = final[benchmark, "fedavg", seed, False]
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "seed": seed,
                        "clean_accuracy_percent": clean[
                            "clean_accuracy_percent"
                        ],
                        "attack_clean_accuracy_percent": attack[
                            "clean_accuracy_percent"
                        ],
                        "clean_loss": clean["clean_loss"],
                        "attack_clean_loss": attack["clean_loss"],
                        "accuracy_damage_pp": clean["clean_accuracy_percent"]
                        - attack["clean_accuracy_percent"],
                        "loss_damage": attack["clean_loss"]
                        - clean["clean_loss"],
                        "clean_asr_percent": clean["asr_percent"],
                        "attack_asr_percent": attack["asr_percent"],
                        "paired_asr_increase_pp": (
                            None
                            if clean["asr_percent"] is None
                            else attack["asr_percent"] - clean["asr_percent"]
                        ),
                        "defense_clean_accuracy_degradation_pp": baseline[
                            "clean_accuracy_percent"
                        ]
                        - clean["clean_accuracy_percent"],
                        "defense_clean_loss_increase": clean["clean_loss"]
                        - baseline["clean_loss"],
                    }
                )
    return rows


def summarize(rows):
    result = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            subset = [
                row
                for row in rows
                if row["benchmark"] == benchmark and row["method"] == method
            ]
            sweep.require(
                len(subset) == 3 and {r["seed"] for r in subset} == set(SEEDS),
                "Missing or duplicate paired seeds",
            )
            record = {
                "benchmark": benchmark,
                "method": method,
                "n_seeds": 3,
                "round": 10,
            }
            for metric in subset[0]:
                if metric in ("benchmark", "method", "seed"):
                    continue
                values = [row[metric] for row in subset]
                sweep.require(
                    all(v is None for v in values)
                    or all(v is not None and np.isfinite(v) for v in values),
                    "Mixed missing or nonfinite metrics",
                )
                record[f"{metric}_mean"] = (
                    None if values[0] is None else statistics.mean(values)
                )
                record[f"{metric}_sample_sd"] = (
                    None if values[0] is None else statistics.stdev(values)
                )
            result.append(record)
    return result


def save_provenance(output, sources, hashes, compatibility):
    source = output / "source"
    source.mkdir()
    for name in SOURCES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    backdoor.write_json(
        output / "config.json",
        {
            "settings": sweep.BASE_SETTINGS,
            "benchmarks": BENCHMARKS,
            "methods": DEFINITIONS,
            "model_poisoning_scale": 10,
            "sybil_backdoor_count": 3,
            "rounds": 10,
            "new_runs": 36,
            "reused_runs": 18,
            "accuracy_damage": "clean minus attack; percentage points",
            "loss_damage": "attack minus clean",
            "paired_asr_increase": "attack minus clean; percentage points",
            "defense_clean_degradation": "FedAvg clean - defense clean",
            "missing_asr": "blank/null for model poisoning; not measured",
        },
    )
    backdoor.write_json(output / "dataset_hashes.json", hashes)
    backdoor.write_json(
        output / "environment.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "flwr")
            },
            "torch_threads": 1,
            "deterministic_algorithms": True,
            "source_sha256": sweep.inventory(source),
        },
    )
    backdoor.write_json(
        output / "provenance.json",
        {
            "reuse": {
                name: {
                    "path": str(path.resolve()),
                    "checksums_sha256": sweep.sha256(path / "checksums.json"),
                    "source_compatibility": compatibility[name],
                }
                for name, path in sources.items()
            },
            "policy": "exact run compatibility; C permits identical AST",
        },
    )
    # Preserve baseline source snapshots and manifests with reuse links.
    for benchmark, origin in sources.items():
        target = output / "baseline_provenance" / benchmark
        target.mkdir(parents=True)
        shutil.copytree(origin / "source", target / "source")
        for name in (
            "checksums.json",
            "environment.json",
            "dataset_hashes.json",
        ):
            shutil.copy2(origin / name, target / name)


def execute(
    output, sources, pair, hashes, expected, configs, baselines, compatibility
):
    output.mkdir(parents=True, exist_ok=False)
    save_provenance(output, sources, hashes, compatibility)
    asr_loader = DataLoader(
        backdoor.asr_dataset(pair[1]), batch_size=32, shuffle=False
    )
    final = {}
    for benchmark in BENCHMARKS:
        for method in METHODS:
            for seed in SEEDS:
                for attack in (False, True):
                    key = benchmark, method, seed, attack
                    folder = (
                        output
                        / benchmark
                        / method
                        / f"seed_{seed}"
                        / ("attack" if attack else "clean")
                    )
                    folder.mkdir(parents=True)
                    backdoor.write_json(folder / "config.json", configs[key])
                    if method == "fedavg":
                        baseline = baselines[benchmark, seed, attack]
                        original = baseline["path"]
                        sweep.require(
                            sweep.inventory(original) == baseline["sha256"],
                            "Baseline changed after preflight",
                        )
                        shutil.copytree(original, folder / "original")
                        backdoor.write_json(
                            folder / "reuse.json",
                            {
                                "path": str(original.resolve()),
                                "sha256": baseline["sha256"],
                                "metrics": baseline["metrics"],
                            },
                        )
                        sweep.require(
                            sweep.inventory(folder / "original")
                            == baseline["sha256"],
                            "Reuse copy differs",
                        )
                        final[key] = baseline["metrics"]
                        continue
                    partitions = expected[seed][0]
                    workers, parts, selected = make_workers(
                        benchmark, pair, partitions, seed, attack
                    )
                    sweep.require(
                        run_config(
                            benchmark,
                            method,
                            workers,
                            parts,
                            selected,
                            seed,
                            attack,
                        )
                        == configs[key],
                        "Preflight identity changed",
                    )
                    np.savez_compressed(
                        folder / "partitions.npz",
                        **{
                            f"client_{i}": np.asarray(part, dtype=np.int64)
                            for i, part in enumerate(parts)
                        },
                    )
                    final[key] = train_condition(
                        workers, benchmark, method, attack, folder, asr_loader
                    )
                    sweep.require(
                        backdoor.dataset_hashes(pair)
                        == hashes["loaded_arrays_sha256"],
                        "Dataset mutated",
                    )
    rows = paired_rows(final)
    sweep.write_csv(output / "per_seed.csv", rows)
    sweep.write_csv(output / "summary.csv", summarize(rows))
    backdoor.write_json(output / "checksums.json", sweep.inventory(output))
    print(f"Defense pilot complete: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start 36 new runs")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sources = {
        "model_poisoning": MODEL_SOURCE,
        "backdoor": sweep.PILOT,
        "sybil_backdoor": SYBIL_SOURCE,
    }
    try:
        sweep.require_fresh_output(args.output)
        sweep.require(not sys.flags.optimize, "Run without -O")
        sweep.require(
            json.loads(json.dumps(asdict(backdoor.SETTINGS)))
            == sweep.BASE_SETTINGS,
            "Pilot settings changed",
        )
        for path in sources.values():
            sweep.require(
                not args.output.resolve().is_relative_to(path.resolve()),
                "Output must not be inside an attack result",
            )
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        pair, hashes = sweep.current_data()
        expected, configs, baselines, compatibility = preflight(
            pair, hashes, sources
        )
        print(json.dumps(DEFINITIONS, indent=2))
        print("3 benchmarks x 3 methods x 3 seeds x 2 conditions = 54 runs.")
        print("Reuse 18 FedAvg runs; 36 new runs / 360 rounds.")
        if not args.run:
            print("Preview passed: no training, downloads, or result writes.")
            return
        execute(
            args.output,
            sources,
            pair,
            hashes,
            expected,
            configs,
            baselines,
            compatibility,
        )
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.exit(1, f"Defense pilot stopped: {error}\n")


if __name__ == "__main__":
    main()
