"""Fixed different-shuffle Sybil robustness test; preview unless --run."""

import argparse
from pathlib import Path
import shutil
import statistics
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_similarity_group as stage

baseline = stage.baseline
pilot = stage.pilot
sweep = stage.sweep
OUTPUT = Path("results/defense_baselines/sybil_distinct_shuffle")
VARIANT = "clip_similarity_group_distinct_shuffle"
RULE = "seed + 1000 + logical_id; reseed only replicas 5 and 6"
EXTRA_SOURCES = (
    "run_sybil_shuffle_robustness.py",
    "tests/test_sybil_shuffle_robustness.py",
    "docs/SYBIL_SHUFFLE_ROBUSTNESS.md",
)


def make_workers(pair, partitions, seed, attack):
    """Only change generator seeds of the two appended replica identities."""
    workers, parts, selected = baseline.make_workers(
        stage.BENCHMARK, pair, partitions, seed, attack
    )
    for logical_id in (5, 6):
        workers[logical_id].trainloader.generator.manual_seed(
            seed + 1000 + logical_id
        )
    return workers, parts, selected


def run_config(workers, parts, selected, seed, attack):
    config = baseline.run_config(
        stage.BENCHMARK, "clip_fedavg", workers, parts, selected, seed, attack
    )
    config.update(method=stage.METHOD, defense_definition=stage.DEFINITION)
    return config


def verify_workers(workers, parts, selected, seed, attack, original):
    config = run_config(workers, parts, selected, seed, attack)
    expected = {
        **original,
        "shuffle_seeds": [seed + 1000 + i for i in range(7)],
    }
    sweep.require(config == expected, "More than shuffle assignments changed")
    attacker_indices = set()
    for logical_id, worker in enumerate(workers):
        generator = worker.trainloader.generator
        expected_generator = torch.Generator().manual_seed(
            seed + 1000 + logical_id
        )
        sweep.require(
            torch.equal(generator.get_state(), expected_generator.get_state()),
            "Incorrect or consumed shuffle stream",
        )
        local = worker.trainset
        if isinstance(local, pilot.TriggeredDataset):
            sweep.require(
                attack
                and logical_id in (0, 5, 6)
                and local.positions == frozenset(selected),
                "Poisoned positions or identities changed",
            )
            local = local.dataset
        else:
            sweep.require(
                not attack or logical_id not in (0, 5, 6),
                "Missing poison transformation",
            )
        sweep.require(
            list(local.indices) == list(parts[logical_id]),
            "Worker partition differs",
        )
        if logical_id in (0, 5, 6):
            sweep.require(
                list(local.indices) == list(parts[0]),
                "Replica received different data",
            )
            attacker_indices.update(local.indices)
    sweep.require(
        len(attacker_indices) == 12000 and attacker_indices == set(parts[0]),
        "Unique attacker data must remain the original 12000",
    )
    return config


def preflight(pair, hashes):
    expected, old_configs, final, _, evidence = stage.preflight(pair, hashes)
    root = stage.OUTPUT
    baseline.verify_archive(root)
    baseline.verify_environment(root)
    sweep.require(
        sweep.read_json(root / "dataset_hashes.json") == hashes,
        "Identical-replica defense dataset differs",
    )
    origins = {}
    configs = {}
    for seed, (partitions, _, _) in expected.items():
        paired = []
        for attack in (False, True):
            mode = "attack" if attack else "clean"
            for method in stage.METHODS:
                folder = root / method / f"seed_{seed}" / mode
                if method == stage.METHOD:
                    sweep.require(
                        sweep.read_json(folder / "config.json")
                        == old_configs[seed, attack],
                        "Identical-replica defense config differs",
                    )
                    with np.load(
                        folder / "partitions.npz", allow_pickle=False
                    ) as saved:
                        mapping = baseline.sybil.partition_mapping(3)
                        sweep.require(
                            set(saved.files)
                            == {f"client_{i}" for i in range(7)},
                            "Wrong partition inventory",
                        )
                        for i, source in enumerate(mapping):
                            sweep.require(
                                np.array_equal(
                                    saved[f"client_{i}"], partitions[source]
                                ),
                                "Identical-replica partition differs",
                            )
                    final[stage.BENCHMARK, method, seed, attack] = (
                        baseline.read_final(stage.BENCHMARK, folder)
                    )
                    for filename, count in (
                        ("aggregation.csv", 70),
                        ("pairwise_similarity.csv", 210),
                        ("grouping_per_round.csv", 10),
                    ):
                        rows = stage.read_rows(folder / filename)
                        sweep.require(
                            len(rows) == count
                            and {int(r["round"]) for r in rows}
                            == set(range(1, 11)),
                            "Incomplete grouping defense logs",
                        )
                else:
                    original = (
                        baseline.OUTPUT
                        / stage.BENCHMARK
                        / method
                        / f"seed_{seed}"
                        / mode
                    )
                    sweep.require(
                        sweep.inventory(folder) == sweep.inventory(original),
                        "Copied comparison baseline differs",
                    )
                origins[stage.BENCHMARK, method, seed, attack] = folder
            workers, parts, selected = make_workers(
                pair, partitions, seed, attack
            )
            config = verify_workers(
                workers,
                parts,
                selected,
                seed,
                attack,
                old_configs[seed, attack],
            )
            configs[seed, attack] = config
            paired.append(pilot.identity(workers, parts))
        sweep.require(paired[0] == paired[1], "Clean/backdoor pair differs")
    rows, summaries = stage.comparison_rows(final)
    stage.validate_table(root / "per_seed.csv", rows)
    stage.validate_table(root / "summary.csv", summaries)
    sweep.require(
        pilot.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
        "Preview changed data",
    )
    return expected, configs, final, origins, evidence


def save_provenance(output, hashes, configs, evidence):
    stage.save_provenance(output, hashes, configs, evidence)
    for name in EXTRA_SOURCES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    environment = sweep.read_json(output / "environment.json")
    environment["source_sha256"] = sweep.inventory(output / "source")
    pilot.write_json(output / "environment.json", environment)
    config = sweep.read_json(output / "config.json")
    config.update(
        experiment="distinct_shuffle_robustness",
        shuffle_rule=RULE,
        reused_runs=24,
        new_runs=6,
        new_rounds=60,
        comparison_label=VARIANT,
        defense_clean_accuracy_degradation_reference=(
            "original identical-replica FedAvg clean; includes shuffle change"
        ),
        matched_shuffle_fedavg_available=False,
    )
    pilot.write_json(output / "config.json", config)
    provenance = sweep.read_json(output / "provenance.json")
    provenance["inputs"].append(
        {
            "path": str(stage.OUTPUT.resolve()),
            "checksums_sha256": sweep.sha256(stage.OUTPUT / "checksums.json"),
        }
    )
    provenance.update(
        shuffle_rule_fixed_before_training=RULE,
        only_changed_setting="generator seeds of logical identities 5 and 6",
        limitation=(
            "Different-shuffle replicas only, not universal robustness; "
            "FedAvg/clip/median comparisons also differ in shuffle scheme"
        ),
    )
    pilot.write_json(output / "provenance.json", provenance)


def train_condition(workers, attack, folder, loader):
    metric, grouping = stage.train_condition(workers, attack, folder, loader)
    # Label only completed diagnostic rows, outside grouping/aggregation.
    rows = stage.read_rows(folder / "pairwise_similarity.csv")
    for row in rows:
        row["pair_type"] = stage.diagnostic.pair_type(
            int(row["client_i"]), int(row["client_j"])
        )
    sweep.write_csv(folder / "pairwise_similarity_labeled.csv", rows)
    return metric, grouping


def summarize_grouping(rows):
    summaries = []
    for condition in ("clean", "attack"):
        subset = [r for r in rows if r["condition"] == condition]
        record = {"condition": condition, "n_seeds": 3}
        for key in subset[0]:
            if key not in ("seed", "condition"):
                values = [r[key] for r in subset]
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_sample_sd"] = statistics.stdev(values)
        summaries.append(record)
    return summaries


def execute(output, pair, hashes, expected, configs, final, origins, evidence):
    output.mkdir(parents=True, exist_ok=False)
    save_provenance(output, hashes, configs, evidence)
    for (_, method, seed, attack), origin in origins.items():
        destination = (
            output
            / method
            / f"seed_{seed}"
            / ("attack" if attack else "clean")
        )
        shutil.copytree(origin, destination)
        sweep.require(
            sweep.inventory(destination) == sweep.inventory(origin),
            "Baseline copy differs",
        )
    loader = DataLoader(
        pilot.asr_dataset(pair[1]), batch_size=32, shuffle=False
    )
    group_rows = []
    for seed, (partitions, _, _) in expected.items():
        for attack in (False, True):
            workers, parts, selected = make_workers(
                pair, partitions, seed, attack
            )
            sweep.require(
                run_config(workers, parts, selected, seed, attack)
                == configs[seed, attack],
                "Preflight drift",
            )
            mode = "attack" if attack else "clean"
            folder = output / VARIANT / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            pilot.write_json(folder / "config.json", configs[seed, attack])
            np.savez_compressed(
                folder / "partitions.npz",
                **{
                    f"client_{i}": np.asarray(part, dtype=np.int64)
                    for i, part in enumerate(parts)
                },
            )
            metric, grouping = train_condition(workers, attack, folder, loader)
            final[stage.BENCHMARK, VARIANT, seed, attack] = metric
            group_rows.append({"seed": seed, "condition": mode, **grouping})
            sweep.require(
                pilot.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
                "Dataset changed during training",
            )
    with patch.multiple(
        baseline,
        BENCHMARKS=(stage.BENCHMARK,),
        METHODS=(*stage.METHODS, VARIANT),
    ):
        rows = baseline.paired_rows(final)
        summary = baseline.summarize(rows)
    sweep.write_csv(output / "per_seed.csv", rows)
    sweep.write_csv(output / "summary.csv", summary)
    sweep.write_csv(output / "grouping_per_seed.csv", group_rows)
    sweep.write_csv(
        output / "grouping_summary.csv", summarize_grouping(group_rows)
    )
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
                "Output cannot be nested in an existing archive",
            )
        sweep.require(not sys.flags.optimize, "Run without -O")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        pair, hashes = sweep.current_data()
        prepared = preflight(pair, hashes)
        print(f"Fixed shuffle rule: {RULE}")
        print("Unchanged cosine threshold: 0.999999; 6 new runs / 60 rounds.")
        print("24 identical-replica comparison runs validated; no retuning.")
        if not args.run:
            print("Preview passed: no training, downloads or result writes.")
            return
        execute(args.output, pair, hashes, *prepared)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.exit(1, f"Shuffle robustness test stopped: {error}\n")


if __name__ == "__main__":
    main()
