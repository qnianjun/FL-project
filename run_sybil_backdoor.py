"""Controlled Sybil-backdoor stage; read-only preview unless --run is given."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_backdoor as pilot
import run_backdoor_sweep as sweep

SYBIL_COUNTS = (1, 2, 3)
OUTPUT = Path("results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2")
EXTRA_SOURCES = (
    "run_backdoor_sweep.py",
    "run_sybil_backdoor.py",
    "tests/test_backdoor_sweep.py",
    "tests/test_sybil_backdoor.py",
)


def partition_mapping(sybil_count):
    """Keep pilot identity order; append client-0 copies."""
    sweep.require(
        type(sybil_count) is int and sybil_count in SYBIL_COUNTS,
        "sybil_count must be 1, 2, or 3",
    )
    return list(range(5)) + [0] * (sybil_count - 1)


def coalition_weight(sybil_count):
    mapping = partition_mapping(sybil_count)
    return mapping.count(0) / len(mapping)


def make_workers(pair, partitions, seed, attack, sybil_count):
    """Independent workers with identical attacker data and shuffle streams."""
    mapping = partition_mapping(sybil_count)
    workers, selected = pilot.make_workers(pair, partitions, seed, attack)
    for _ in range(sybil_count - 1):
        worker = pilot.BalancedClient(0, pair, partitions, seed)
        if attack:
            worker.trainset = pilot.TriggeredDataset(worker.trainset, selected)
            worker.trainloader = DataLoader(
                worker.trainset,
                batch_size=pilot.SETTINGS.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + 1000),
            )
        workers.append(worker)
    for logical_id, worker in enumerate(workers):
        # cid is the original partition; logical_id is the server identity.
        worker.logical_id = logical_id
    logical_parts = [partitions[index] for index in mapping]
    return workers, logical_parts, selected


def verify_workers(workers, partitions, selected, attack, sybil_count):
    """Verify duplication without consuming the training shuffle generators."""
    mapping = partition_mapping(sybil_count)
    sweep.require(
        len(workers) == len(mapping), "Logical identity count changed"
    )
    attacker_indices = set()
    for logical_id, (worker, original_id) in enumerate(zip(workers, mapping)):
        sweep.require(worker.logical_id == logical_id, "Duplicate identity ID")
        sweep.require(worker.cid == original_id, "Wrong original partition")
        dataset = worker.trainset
        poisoned = attack and original_id == 0
        sweep.require(
            isinstance(dataset, pilot.TriggeredDataset) == poisoned,
            "Poisoning applied to the wrong identity",
        )
        if poisoned:
            sweep.require(
                dataset.positions == frozenset(selected),
                "Sybil poison selection differs",
            )
            dataset = dataset.dataset
        sweep.require(
            list(dataset.indices) == partitions[original_id],
            "Sybil received different or additional data",
        )
        if original_id == 0:
            attacker_indices.update(dataset.indices)
            sweep.require(
                torch.equal(
                    worker.trainloader.generator.get_state(),
                    workers[0].trainloader.generator.get_state(),
                ),
                "Attacker shuffle streams differ",
            )
    sweep.require(
        attacker_indices == set(partitions[0]), "Unique attacker data changed"
    )
    reported = sum(len(worker.trainset) for worker in workers)
    malicious = sum(
        len(worker.trainset) for worker in workers if worker.cid == 0
    )
    sweep.require(
        malicious / reported == coalition_weight(sybil_count),
        "Reported aggregation weight differs from design",
    )


def condition_config(workers, partitions, selected, seed, attack, count):
    mapping = partition_mapping(count)
    logical_parts = [partitions[index] for index in mapping]
    return {
        **sweep.BASE_SETTINGS,
        "num_clients": len(mapping),
        "original_num_partitions": 5,
        "sybil_count": count,
        "seed": seed,
        "condition": "backdoor" if attack else "clean",
        "logical_to_original_partition": mapping,
        "attacker_logical_ids": [
            i for i, source in enumerate(mapping) if source == 0
        ],
        "unique_attacker_sample_count": len(set(partitions[0])),
        "unique_total_sample_count": len(set(np.concatenate(partitions))),
        "reported_total_sample_count": sum(len(w.trainset) for w in workers),
        "attacker_coalition_weight": coalition_weight(count),
        "poison_selection_seed": seed + 2000,
        "poisoned_local_positions_per_attacker": (
            selected.tolist() if attack else []
        ),
        "asr_sample_count": 9020,
        **pilot.identity(workers, logical_parts),
    }


def preflight(pair, hashes, pilot_path):
    expected = sweep.expected_pairs(pair, 0.1)
    final_pilot = sweep.validate_result(
        pilot_path, 0.1, pair, hashes, expected
    )
    asr_set = pilot.asr_dataset(pair[1])
    eligible = np.flatnonzero(np.asarray(pair[1].targets) != 0)
    sweep.require(
        len(asr_set) == len(eligible) == 9020
        and np.array_equal(asr_set.dataset.indices, eligible),
        "ASR denominator or eligible indices changed",
    )
    configs = {}
    for seed, (partitions, base_identity, selected) in expected.items():
        for count in SYBIL_COUNTS:
            identities = []
            for attack in (False, True):
                workers, logical_parts, repeated = make_workers(
                    pair, partitions, seed, attack, count
                )
                sweep.require(
                    np.array_equal(selected, repeated), "Selection changed"
                )
                verify_workers(workers, partitions, repeated, attack, count)
                # Audit every original local sample once per seed/condition.
                # Replicas reuse audited data, transformation and positions.
                if count == 1:
                    pilot.verify_transforms(workers, pair, repeated, attack)
                current_identity = pilot.identity(workers, logical_parts)
                if count == 1:
                    sweep.require(
                        current_identity == base_identity, "Pilot differs"
                    )
                identities.append(current_identity)
                configs[count, seed, attack] = condition_config(
                    workers, partitions, repeated, seed, attack, count
                )
            sweep.require(
                identities[0] == identities[1], "Pair is not matched"
            )
            print(
                f"seed={seed} sybil_count={count}: matched; "
                f"identities={4 + count}; unique attacker samples=12000; "
                f"coalition weight={coalition_weight(count):.6%}",
                flush=True,
            )
    sweep.require(
        pilot.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
        "Underlying MNIST data changed during validation",
    )
    return expected, configs, final_pilot


def paired_rows(count, final):
    rows = sweep.paired_rows(0.1, final)
    for row in rows:
        row["sybil_count"] = count
        row["logical_identity_count"] = 4 + count
        row["attacker_coalition_weight"] = coalition_weight(count)
        pair = {r["condition"]: r for r in final if r["seed"] == row["seed"]}
        row["clean_loss"] = pair["clean"]["clean_loss"]
        row["backdoor_clean_loss"] = pair["backdoor"]["clean_loss"]
        row["paired_clean_loss_change"] = (
            row["backdoor_clean_loss"] - row["clean_loss"]
        )
    return rows


def summary_rows(rows):
    result = []
    fixed = ("alpha", "seed", "round", "sybil_count", "logical_identity_count")
    for count in SYBIL_COUNTS:
        subset = [row for row in rows if row["sybil_count"] == count]
        sweep.require(
            len(subset) == 3 and {r["seed"] for r in subset} == {42, 43, 44},
            f"Incomplete or duplicate seeds for sybil_count={count}",
        )
        summary = {
            "sybil_count": count,
            "alpha": 0.1,
            "n_seeds": 3,
            "round": 10,
            "logical_identity_count": 4 + count,
        }
        for metric in subset[0]:
            if metric not in fixed:
                values = [row[metric] for row in subset]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sample_sd"] = statistics.stdev(values)
        result.append(summary)
    return result


def validate_new_result(folder, count, expected, configs):
    names = {"config.json", "summary.json"}
    for seed in pilot.SETTINGS.seeds:
        for mode in ("clean", "backdoor"):
            names.update(
                f"seed_{seed}/{mode}/{name}"
                for name in ("config.json", "partitions.npz", "results.csv")
            )
    manifest = sweep.read_json(folder / "checksums.json")
    sweep.require(set(manifest) == names, "Incomplete result inventory")
    sweep.require(sweep.inventory(folder) == manifest, "Result hashes differ")
    sweep.require(
        sweep.read_json(folder / "config.json")
        == {
            "sybil_count": count,
            "attacker_coalition_weight": coalition_weight(count),
        },
        "Setting config mismatch",
    )
    final = []
    for seed, (partitions, _, _) in expected.items():
        for attack in (False, True):
            mode = "backdoor" if attack else "clean"
            local = folder / f"seed_{seed}" / mode
            sweep.require(
                sweep.read_json(local / "config.json")
                == configs[count, seed, attack],
                "Saved condition differs from preflight",
            )
            with np.load(
                local / "partitions.npz", allow_pickle=False
            ) as saved:
                mapping = partition_mapping(count)
                sweep.require(
                    set(saved.files)
                    == {f"client_{i}" for i in range(len(mapping))},
                    "Saved logical identities differ",
                )
                for logical_id, original_id in enumerate(mapping):
                    indices = saved[f"client_{logical_id}"]
                    sweep.require(
                        indices.dtype == np.dtype("int64")
                        and np.array_equal(indices, partitions[original_id]),
                        "Saved Sybil partition differs",
                    )
            final.append(
                {
                    "seed": seed,
                    "condition": mode,
                    **sweep.read_metrics(local / "results.csv"),
                }
            )
    sweep.require(
        sweep.read_json(folder / "summary.json") == pilot.summarize(final),
        "Saved metrics summary differs",
    )
    return final


def train_setting(folder, count, pair, hashes, expected, configs):
    folder.mkdir()
    pilot.write_json(
        folder / "config.json",
        {
            "sybil_count": count,
            "attacker_coalition_weight": coalition_weight(count),
        },
    )
    loader = DataLoader(
        pilot.asr_dataset(pair[1]), batch_size=32, shuffle=False
    )
    final = []
    for seed, (partitions, _, _) in expected.items():
        for attack in (False, True):
            mode = "backdoor" if attack else "clean"
            workers, logical_parts, selected = make_workers(
                pair, partitions, seed, attack, count
            )
            verify_workers(workers, partitions, selected, attack, count)
            config = condition_config(
                workers, partitions, selected, seed, attack, count
            )
            sweep.require(
                config == configs[count, seed, attack], "Preflight drift"
            )
            local = folder / f"seed_{seed}" / mode
            local.mkdir(parents=True)
            pilot.write_json(local / "config.json", config)
            np.savez_compressed(
                local / "partitions.npz",
                **{
                    f"client_{i}": np.asarray(part, dtype=np.int64)
                    for i, part in enumerate(logical_parts)
                },
            )
            final.append(
                pilot.train_condition(workers, loader, local, seed, mode)
            )
            sweep.require(
                pilot.dataset_hashes(pair) == hashes["loaded_arrays_sha256"],
                "Underlying MNIST changed during training",
            )
    pilot.write_json(folder / "summary.json", pilot.summarize(final))
    pilot.write_json(folder / "checksums.json", sweep.inventory(folder))
    return validate_new_result(folder, count, expected, configs)


def save_provenance(output, pilot_path, hashes, configs):
    source = output / "source"
    source.mkdir()
    for name in (*sweep.SOURCES, *EXTRA_SOURCES):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    environment = sweep.read_json(pilot_path / "environment.json")
    # Preflight checked versions against the current environment.
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
            "original_partition_settings": sweep.BASE_SETTINGS,
            "sybil_counts": SYBIL_COUNTS,
            "logical_to_original_partition": {
                count: partition_mapping(count) for count in SYBIL_COUNTS
            },
            "aggregation": "sample-weighted FedAvg; 12000 per identity",
            "attacker_shuffle": "all attacker identities use seed + 1000",
            "honest_shuffle": "seed + 1000 + original partition ID",
            "poison_selection": "same 2400 positions per Sybil; seed + 2000",
            "clean_control": "same duplication without poisoning",
            "asr_denominator": "9020 triggered images; original label != 0",
            "paired_difference": "backdoor minus clean; percentage points",
        },
    )
    pilot.write_json(
        output / "provenance.json",
        {
            "pilot_path": str(pilot_path.resolve()),
            "pilot_checksums_sha256": sweep.sha256(
                pilot_path / "checksums.json"
            ),
            "reuse_policy": "exact source, versions, data, configs, metrics",
            "reused_runs": 6,
            "new_runs": 12,
            "preflight_conditions": [configs[key] for key in sorted(configs)],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start 12 new runs")
    parser.add_argument("--pilot", type=Path, default=sweep.PILOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        sweep.require_fresh_output(args.output)
        sweep.require(
            not args.output.resolve().is_relative_to(args.pilot.resolve()),
            "Output must not be inside the existing pilot",
        )
        sweep.require(not sys.flags.optimize, "Run without -O")
        sweep.require(
            json.loads(json.dumps(asdict(pilot.SETTINGS)))
            == sweep.BASE_SETTINGS,
            "Validated pilot settings changed",
        )
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        pair, hashes = sweep.current_data()
        expected, configs, final_pilot = preflight(pair, hashes, args.pilot)
        print("Exact pilot reuse passed: sybil_count=1, six completed runs.")
        print("12 new runs (120 rounds); ASR denominator=9020.")
        if not args.run:
            print("Preview passed. No training, downloads, or result writes.")
            return
        args.output.mkdir(parents=True, exist_ok=False)
        save_provenance(args.output, args.pilot, hashes, configs)
        reused = args.output / "sybil_1"
        shutil.copytree(args.pilot, reused)
        final = sweep.validate_result(reused, 0.1, pair, hashes, expected)
        sweep.require(final == final_pilot, "Pilot changed after preflight")
        rows = paired_rows(1, final)
        for count in (2, 3):
            final = train_setting(
                args.output / f"sybil_{count}",
                count,
                pair,
                hashes,
                expected,
                configs,
            )
            rows.extend(paired_rows(count, final))
        sweep.write_csv(args.output / "per_seed.csv", rows)
        sweep.write_csv(args.output / "summary.csv", summary_rows(rows))
        pilot.write_json(
            args.output / "checksums.json", sweep.inventory(args.output)
        )
        print(f"Sybil-backdoor stage complete: {args.output}")
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.exit(1, f"Sybil stage stopped: {error}\n")


if __name__ == "__main__":
    main()
