"""Exploratory identical-replica Sybil defense; preview unless --run."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_defense_pilot as baseline
import run_update_similarity as diagnostic
from similarity_group_aggregation import (
    DEFINITION,
    METHOD,
    aggregate_grouped,
    analyze_groups,
)

sweep = baseline.sweep
pilot = baseline.backdoor
BENCHMARK = "sybil_backdoor"
METHODS = (*baseline.METHODS, METHOD)
OUTPUT = Path("results/defense_baselines/clip_similarity_group_pilot")
SOURCES = (
    *baseline.SOURCES,
    "run_update_similarity.py",
    "tests/test_update_similarity.py",
    "similarity_group_aggregation.py",
    "run_similarity_group.py",
    "tests/test_similarity_group.py",
    "docs/SIMILARITY_GROUP_PILOT.md",
)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def comparison_rows(final):
    with patch.multiple(baseline, BENCHMARKS=(BENCHMARK,), METHODS=METHODS):
        rows = baseline.paired_rows(final)
        return rows, baseline.summarize(rows)


def validate_table(path, expected):
    """Recompute reused summary values from per-condition raw metrics."""
    saved = [r for r in read_rows(path) if r["benchmark"] == BENCHMARK]
    keys = (
        ("benchmark", "method", "seed")
        if "seed" in expected[0]
        else ("benchmark", "method")
    )
    by_key = {tuple(row[k] for k in keys): row for row in saved}
    sweep.require(
        len(saved) == len(by_key) == len(expected),
        "Missing or duplicate baseline summary rows",
    )
    for row in expected:
        actual = by_key[tuple(str(row[k]) for k in keys)]
        sweep.require(set(actual) == set(row), "Summary fields differ")
        for key, value in row.items():
            if isinstance(value, (int, float)):
                sweep.require(
                    np.isclose(
                        float(actual[key]), value, rtol=1e-12, atol=1e-12
                    ),
                    f"Baseline summary differs: {key}",
                )
            else:
                sweep.require(actual[key] == str(value), "Summary differs")


def validate_diagnostic(configs):
    root = diagnostic.OUTPUT
    baseline.verify_archive(root)
    baseline.verify_environment(root)
    provenance = sweep.read_json(root / "provenance.json")
    sweep.require(
        provenance["source_manifest_sha256"]
        == sweep.sha256(baseline.sybil.OUTPUT / "checksums.json"),
        "Diagnostic origin differs",
    )
    values = {}
    for seed in baseline.SEEDS:
        for attack in (False, True):
            mode = "backdoor" if attack else "clean"
            folder = root / f"seed_{seed}" / mode
            sweep.require(
                sweep.read_json(folder / "config.json")
                == configs[3, seed, attack],
                "Diagnostic config differs",
            )
            original = (
                baseline.sybil.OUTPUT / "sybil_3" / f"seed_{seed}" / mode
            )
            for name in ("results.csv", "partitions.npz"):
                sweep.require(
                    sweep.sha256(folder / name)
                    == sweep.sha256(original / name),
                    "Diagnostic trajectory/partition differs",
                )
            pairs = read_rows(folder / "pairwise_similarity.csv")
            expected = {
                (r, i, j)
                for r in range(1, 11)
                for i in range(7)
                for j in range(i + 1, 7)
            }
            actual = {
                (int(r["round"]), int(r["client_i"]), int(r["client_j"]))
                for r in pairs
            }
            sweep.require(
                len(pairs) == 210 and actual == expected,
                "Incomplete diagnostic pair coverage",
            )
            for row in pairs:
                label = diagnostic.pair_type(
                    int(row["client_i"]), int(row["client_j"])
                )
                sweep.require(row["pair_type"] == label, "Wrong pair label")
                if row["cosine_similarity"]:
                    value = float(row["cosine_similarity"])
                    sweep.require(
                        np.isfinite(value) and -1 <= value <= 1,
                        "Invalid diagnostic cosine",
                    )
                    values.setdefault(label, []).append(value)
    return {
        key: {"count": len(v), "min": min(v), "max": max(v)}
        for key, v in values.items()
    }


def preflight(pair, hashes):
    expected, sybil_configs = diagnostic.preflight(pair, hashes)
    evidence = validate_diagnostic(sybil_configs)
    root = baseline.OUTPUT
    baseline.verify_archive(root)
    baseline.verify_environment(root)
    sweep.require(
        sweep.read_json(root / "dataset_hashes.json") == hashes,
        "Defense baseline dataset differs",
    )
    final = {}
    configs = {}
    origins = {}
    for seed, (partitions, _, _) in expected.items():
        for attack in (False, True):
            workers, parts, selected = baseline.make_workers(
                BENCHMARK, pair, partitions, seed, attack
            )
            for method in baseline.METHODS:
                config = baseline.run_config(
                    BENCHMARK, method, workers, parts, selected, seed, attack
                )
                folder = (
                    root
                    / BENCHMARK
                    / method
                    / f"seed_{seed}"
                    / ("attack" if attack else "clean")
                )
                sweep.require(
                    sweep.read_json(folder / "config.json") == config,
                    "Incompatible baseline condition",
                )
                metrics_folder = (
                    folder / "original" if method == "fedavg" else folder
                )
                with np.load(
                    metrics_folder / "partitions.npz", allow_pickle=False
                ) as saved:
                    sweep.require(
                        set(saved.files) == {f"client_{i}" for i in range(7)},
                        "Wrong baseline partition inventory",
                    )
                    for i, part in enumerate(parts):
                        sweep.require(
                            np.array_equal(saved[f"client_{i}"], part),
                            "Baseline partition differs",
                        )
                if method == "fedavg":
                    mode = "backdoor" if attack else "clean"
                    original = (
                        baseline.sybil.OUTPUT
                        / "sybil_3"
                        / f"seed_{seed}"
                        / mode
                    )
                    sweep.require(
                        sweep.inventory(metrics_folder)
                        == sweep.inventory(original),
                        "Reused FedAvg differs",
                    )
                else:
                    logs = read_rows(folder / "aggregation.csv")
                    sweep.require(
                        len(logs) == 70
                        and {
                            (int(r["round"]), int(r["logical_id"]))
                            for r in logs
                        }
                        == {(r, i) for r in range(1, 11) for i in range(7)},
                        "Incomplete baseline aggregation log",
                    )
                key = BENCHMARK, method, seed, attack
                final[key] = baseline.read_final(BENCHMARK, metrics_folder)
                origins[key] = folder
            config = baseline.run_config(
                BENCHMARK,
                "clip_fedavg",
                workers,
                parts,
                selected,
                seed,
                attack,
            )
            config.update(method=METHOD, defense_definition=DEFINITION)
            configs[seed, attack] = config
    with patch.multiple(baseline, BENCHMARKS=(BENCHMARK,)):
        rows = baseline.paired_rows(final)
        summaries = baseline.summarize(rows)
    validate_table(root / "per_seed.csv", rows)
    validate_table(root / "summary.csv", summaries)
    return expected, configs, final, origins, evidence


def train_condition(workers, attack, folder, loader):
    rounds = []
    groups_output = []
    pairs_output = []

    def aggregate(parameters, submissions, method):
        sweep.require(method == METHOD, "Unexpected aggregation method")
        result, records, groups, pairs = aggregate_grouped(
            parameters, submissions
        )
        number = len(rounds) + 1
        # Labels are introduced only after the aggregate result is computed.
        rounds.append({"round": number, **analyze_groups(groups, {0, 5, 6})})
        for group_id, members in enumerate(groups):
            groups_output.append(
                {
                    "round": number,
                    "group_id": group_id,
                    "members": json.dumps(members),
                    "group_size": len(members),
                    "group_weight": 1 / len(groups),
                    "per_identity_weight": 1 / len(groups) / len(members),
                }
            )
        for i, j, cosine in pairs:
            pairs_output.append(
                {
                    "round": number,
                    "client_i": i,
                    "client_j": j,
                    "cosine_similarity": cosine,
                }
            )
        return result, records

    with patch.object(baseline, "aggregate_updates", aggregate):
        final = baseline.train_condition(
            workers, BENCHMARK, METHOD, attack, folder, loader
        )
    sweep.write_csv(folder / "groups.csv", groups_output)
    sweep.write_csv(folder / "pairwise_similarity.csv", pairs_output)
    sweep.write_csv(folder / "grouping_per_round.csv", rounds)
    return final, {
        key: statistics.mean(row[key] for row in rounds)
        for key in rounds[0]
        if key != "round"
    }


def save_provenance(output, hashes, configs, evidence):
    source = output / "source"
    source.mkdir()
    for name in SOURCES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    environment = sweep.read_json(baseline.OUTPUT / "environment.json")
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
            "method": METHOD,
            "definition": DEFINITION,
            "conditions": [configs[k] for k in sorted(configs)],
            "new_runs": 6,
            "reused_runs": 18,
            "new_rounds": 60,
            "analysis_weight": (
                "sum_g (replica members / group size) / group count; "
                "before clip factors"
            ),
            "clean_replica_labels": (
                "structural partition-0 replicas, not malicious clean updates"
            ),
            "false_grouping": (
                "honest identities in nonsingletons, grouped honest pairs, "
                "mixed groups"
            ),
            "seed_summary": (
                "sample SD across 3 seeds; "
                "grouping uses each seed's round mean"
            ),
        },
    )
    pilot.write_json(
        output / "provenance.json",
        {
            "inputs": [
                {
                    "path": str(root.resolve()),
                    "checksums_sha256": sweep.sha256(root / "checksums.json"),
                }
                for root in (
                    baseline.OUTPUT,
                    diagnostic.OUTPUT,
                    baseline.sybil.OUTPUT,
                )
            ],
            "compatibility": (
                "exact source/environment/configs/partitions "
                "and verified raw metrics"
            ),
            "diagnostic_evidence": evidence,
            "criterion_fixed_before_defense_training": DEFINITION,
            "limitation": (
                "exploratory identical replicas only; "
                "no universal detection claim"
            ),
        },
    )


def execute(output, pair, hashes, expected, configs, final, origins, evidence):
    output.mkdir(parents=True, exist_ok=False)
    save_provenance(output, hashes, configs, evidence)
    for key, origin in origins.items():
        _, method, seed, attack = key
        target = (
            output
            / method
            / f"seed_{seed}"
            / ("attack" if attack else "clean")
        )
        shutil.copytree(origin, target)
        sweep.require(
            sweep.inventory(target) == sweep.inventory(origin),
            "Baseline copy differs",
        )
    loader = DataLoader(
        pilot.asr_dataset(pair[1]), batch_size=32, shuffle=False
    )
    group_rows = []
    for seed, (partitions, _, _) in expected.items():
        for attack in (False, True):
            workers, parts, selected = baseline.make_workers(
                BENCHMARK, pair, partitions, seed, attack
            )
            config = baseline.run_config(
                BENCHMARK,
                "clip_fedavg",
                workers,
                parts,
                selected,
                seed,
                attack,
            )
            config.update(method=METHOD, defense_definition=DEFINITION)
            sweep.require(config == configs[seed, attack], "Preflight drift")
            mode = "attack" if attack else "clean"
            folder = output / METHOD / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            pilot.write_json(folder / "config.json", config)
            np.savez_compressed(
                folder / "partitions.npz",
                **{
                    f"client_{i}": np.asarray(part, dtype=np.int64)
                    for i, part in enumerate(parts)
                },
            )
            metric, grouping = train_condition(workers, attack, folder, loader)
            final[BENCHMARK, METHOD, seed, attack] = metric
            group_rows.append({"seed": seed, "condition": mode, **grouping})
            sweep.require(
                pilot.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
                "Dataset changed",
            )
    rows, summary = comparison_rows(final)
    sweep.write_csv(output / "per_seed.csv", rows)
    sweep.write_csv(output / "summary.csv", summary)
    sweep.write_csv(output / "grouping_per_seed.csv", group_rows)
    summaries = []
    for mode in ("clean", "attack"):
        subset = [r for r in group_rows if r["condition"] == mode]
        record = {"condition": mode, "n_seeds": 3}
        for key in subset[0]:
            if key not in ("seed", "condition"):
                values = [r[key] for r in subset]
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_sample_sd"] = statistics.stdev(values)
        summaries.append(record)
    sweep.write_csv(output / "grouping_summary.csv", summaries)
    pilot.write_json(output / "checksums.json", sweep.inventory(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        sweep.require_fresh_output(args.output)
        for ancestor in args.output.resolve().parents:
            sweep.require(
                not (ancestor / "checksums.json").exists(),
                "Output cannot be nested inside an archived result",
            )
        for root in (
            baseline.OUTPUT,
            diagnostic.OUTPUT,
            baseline.sybil.OUTPUT,
            sweep.PILOT,
        ):
            sweep.require(
                not args.output.resolve().is_relative_to(root.resolve()),
                "Output cannot be inside prior results",
            )
        sweep.require(not sys.flags.optimize, "Run without -O")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        pair, hashes = sweep.current_data()
        expected, configs, final, origins, evidence = preflight(pair, hashes)
        print(json.dumps(DEFINITION, indent=2))
        print("Validated reuse: 18 runs; new: 6 matched runs / 60 rounds.")
        if not args.run:
            print("Preview passed. No training, downloads or result writes.")
            return
        execute(
            args.output,
            pair,
            hashes,
            expected,
            configs,
            final,
            origins,
            evidence,
        )
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.exit(1, f"Similarity-group pilot stopped: {error}\n")


if __name__ == "__main__":
    main()
