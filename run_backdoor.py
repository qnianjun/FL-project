"""Matched MNIST backdoor pilot. Default: validate only; --run trains."""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import DataLoader, Dataset, Subset

import client
from balanced_partition import balanced_partition
from run_balanced import BalancedClient, digest_arrays, write_json


@dataclass(frozen=True)
class Settings:
    alpha: float = 0.1
    poison_ratio: float = 0.2
    target_label: int = 0
    trigger_size: int = 3
    trigger_row: int = 25
    trigger_column: int = 25
    trigger_value: float = 1.0
    seeds: tuple = (42, 43, 44)
    num_clients: int = 5
    attacker: int = 0
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "SGD"


SETTINGS = Settings()


def apply_trigger(image, settings=SETTINGS):
    """Return a triggered copy of a ToTensor MNIST image."""
    result = image.clone()
    row = settings.trigger_row
    column = settings.trigger_column
    size = settings.trigger_size
    result[:, row:row + size, column:column + size] = settings.trigger_value
    return result


def poison_selection(count, ratio, seed):
    """Select floor(ratio * count) local positions once, without replacement.

    The denominator is all attacker samples, including original target labels.
    Selection stays fixed across rounds and uses an independent local RNG.
    """
    if not 0 <= ratio <= 1:
        raise ValueError("poison_ratio must be between zero and one")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, int(count * ratio), replace=False))


class TriggeredDataset(Dataset):
    """Lazy copy-on-read transformation; never mutate the underlying dataset."""

    def __init__(self, dataset, positions, settings=SETTINGS):
        self.dataset = dataset
        self.positions = frozenset(int(index) for index in positions)
        self.settings = settings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if index in self.positions:
            return apply_trigger(image, self.settings), self.settings.target_label
        return image, label


def make_workers(datasets_pair, partitions, seed, attack):
    workers = [
        BalancedClient(cid, datasets_pair, partitions, seed)
        for cid in range(SETTINGS.num_clients)
    ]
    selected = poison_selection(
        len(partitions[SETTINGS.attacker]), SETTINGS.poison_ratio, seed + 2000
    )
    if attack:
        worker = workers[SETTINGS.attacker]
        worker.trainset = TriggeredDataset(worker.trainset, selected)
        worker.trainloader = DataLoader(
            worker.trainset,
            batch_size=SETTINGS.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(
                seed + 1000 + SETTINGS.attacker
            ),
        )
    return workers, selected


def identity(workers, partitions):
    initial_hashes = [
        digest_arrays(worker.get_parameters({})) for worker in workers
    ]
    assert len(set(initial_hashes)) == 1
    actual_indices = []
    for worker, expected in zip(workers, partitions):
        local = worker.trainset
        if isinstance(local, TriggeredDataset):
            local = local.dataset
        assert list(local.indices) == list(expected)
        actual_indices.append(np.asarray(local.indices, dtype=np.int64))
    return {
        "initial_parameters_sha256": initial_hashes[0],
        "partition_sha256": [digest_arrays([p]) for p in actual_indices],
        "shuffle_seeds": [
            worker.trainloader.generator.initial_seed() for worker in workers
        ],
        "client_sample_counts": [len(worker.trainset) for worker in workers],
    }


def verify_transforms(workers, datasets_pair, selected, attack):
    """Audit every local sample without training, including honest clients."""
    for worker in workers:
        local = worker.trainset
        poisoned = attack and worker.cid == SETTINGS.attacker
        assert isinstance(local, TriggeredDataset) == poisoned
        original = local.dataset if poisoned else local
        assert worker.testset is datasets_pair[1]
        assert worker.testloader.dataset is datasets_pair[1]
        selected_positions = set(selected) if poisoned else set()
        for position in range(len(local)):
            image, label = original[position]
            actual_image, actual_label = local[position]
            if position in selected_positions:
                assert actual_label == SETTINGS.target_label
                assert torch.equal(actual_image, apply_trigger(image))
            else:
                assert int(actual_label) == int(label)
                assert torch.equal(actual_image, image)


def asr_dataset(testset):
    indices = np.flatnonzero(
        np.asarray(testset.targets) != SETTINGS.target_label
    )
    eligible = Subset(testset, indices.tolist())
    assert len(eligible) > 0
    return TriggeredDataset(eligible, range(len(eligible)))


def dataset_hashes(datasets_pair):
    return {
        name: digest_arrays([dataset.data, dataset.targets])
        for name, dataset in zip(("train", "test"), datasets_pair)
    }


def summarize(rows):
    metrics = ("clean_loss", "clean_accuracy_percent", "asr_percent")
    summary = {"final_round_per_seed": rows, "conditions": {}}
    for mode in ("clean", "backdoor"):
        subset = [row for row in rows if row["condition"] == mode]
        summary["conditions"][mode] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in subset),
                "sample_std": statistics.stdev(row[metric] for row in subset),
            }
            for metric in metrics
        }
    differences = []
    for seed in SETTINGS.seeds:
        paired = {row["condition"]: row for row in rows if row["seed"] == seed}
        differences.append({
            "seed": seed,
            **{
                metric: paired["backdoor"][metric] - paired["clean"][metric]
                for metric in metrics
            },
        })
    summary["paired_backdoor_minus_clean"] = {
        "units": "percentage points for accuracy/ASR; loss units for loss",
        "per_seed": differences,
        "statistics": {
            metric: {
                "mean": statistics.mean(row[metric] for row in differences),
                "sample_std": statistics.stdev(row[metric] for row in differences),
            }
            for metric in metrics
        },
    }
    return summary


def train_condition(workers, asr_loader, folder, seed, mode):
    parameters = workers[0].get_parameters({})
    with (folder / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "round", "clean_loss", "clean_accuracy_percent", "asr_percent"
        ])
        for round_number in range(1, SETTINGS.rounds + 1):
            updates = []
            for worker in workers:
                worker.set_parameters(parameters)
                # Reuse SGD/one-epoch training directly: no update scaling.
                client.train(worker.model, worker.trainloader)
                updates.append((worker.get_parameters({}), len(worker.trainset)))
            parameters = aggregate(updates)
            worker = workers[0]
            worker.set_parameters(parameters)
            loss, accuracy = client.test(worker.model, worker.testloader)
            _, asr = client.test(worker.model, asr_loader)
            assert np.isfinite(loss) and 0 <= accuracy <= 1 and 0 <= asr <= 1
            writer.writerow([round_number, loss, accuracy * 100, asr * 100])
            handle.flush()
            print(
                f"seed={seed} {mode} round={round_number} "
                f"accuracy={accuracy:.4f} ASR={asr:.4f}",
                flush=True,
            )
    return {
        "seed": seed,
        "condition": mode,
        "round": SETTINGS.rounds,
        "clean_loss": loss,
        "clean_accuracy_percent": accuracy * 100,
        "asr_percent": asr * 100,
    }


def save_provenance(output, hashes):
    source = output / "source"
    source.mkdir()
    for name in (
        "run_backdoor.py", "client.py", "run_balanced.py",
        "balanced_partition.py", "config.json", "tests/test_backdoor.py",
    ):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    write_json(output / "config.json", {
        **asdict(SETTINGS),
        "partition_method": "capacity_constrained_dirichlet_v1",
        "aggregation": "sample-weighted FedAvg; equal counts give 20% each",
        "poison_selection": "floor(ratio * all attacker samples), fixed per seed",
        "poison_selection_seed": "seed + 2000",
        "shuffle_seed": "seed + 1000 + client_id",
        "asr_denominator": "test images with original label != target_label",
        "execution": "sequential CPU; all clients every round; no update scaling",
    })
    write_json(output / "dataset_hashes.json", {
        "loaded_arrays_sha256": hashes,
        "raw_files_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("data/MNIST/raw").iterdir())
            if path.is_file() and path.suffix != ".gz"
        },
    })
    write_json(output / "environment.json", {
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
        "source_sha256": {
            str(path.relative_to(source)):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*")) if path.is_file()
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start pilot training")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/Backdoor_test/mnist_alpha_0.1_ratio_0.2"),
    )
    args = parser.parse_args()
    if args.output.exists() and args.run:
        parser.error("Output already exists; choose a fresh --output directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets_pair = client.load_data()
    original_hashes = dataset_hashes(datasets_pair)
    triggered_test = asr_dataset(datasets_pair[1])
    asr_loader = DataLoader(triggered_test, batch_size=32, shuffle=False)
    print(json.dumps(asdict(SETTINGS), indent=2))
    if args.run:
        args.output.mkdir(parents=True, exist_ok=False)
        save_provenance(args.output, original_hashes)
    final = []
    for seed in SETTINGS.seeds:
        partitions = balanced_partition(
            datasets_pair[0].targets, SETTINGS.num_clients, SETTINGS.alpha, seed
        )
        assert sorted(index for part in partitions for index in part) == list(
            range(len(datasets_pair[0]))
        )
        assert len(set(map(len, partitions))) == 1
        paired_identity = None
        for attack in (False, True):
            mode = "backdoor" if attack else "clean"
            workers, selected = make_workers(datasets_pair, partitions, seed, attack)
            current_identity = identity(workers, partitions)
            if paired_identity is None:
                paired_identity = current_identity
            else:
                assert current_identity == paired_identity, "Pair is not matched"
            verify_transforms(workers, datasets_pair, selected, attack)
            assert dataset_hashes(datasets_pair) == original_hashes
            print(
                f"seed={seed} {mode}: validation passed; "
                f"counts={current_identity['client_sample_counts']}; "
                f"poisoned={len(selected) if attack else 0}; "
                f"ASR denominator={len(triggered_test)}",
                flush=True,
            )
            if not args.run:
                continue
            folder = args.output / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            np.savez_compressed(
                folder / "partitions.npz",
                **{
                    f"client_{cid}": np.asarray(part, dtype=np.int64)
                    for cid, part in enumerate(partitions)
                },
            )
            write_json(folder / "config.json", {
                **asdict(SETTINGS), **current_identity,
                "seed": seed, "condition": mode,
                "poison_selection_seed": seed + 2000,
                "poisoned_local_positions": selected.tolist() if attack else [],
                "client_label_counts": [
                    np.bincount(
                        np.asarray(datasets_pair[0].targets)[part], minlength=10
                    ).tolist()
                    for part in partitions
                ],
                "asr_sample_count": len(triggered_test),
                "validation": "matched identity, transforms, dataset hashes passed",
            })
            final.append(train_condition(workers, asr_loader, folder, seed, mode))
            assert dataset_hashes(datasets_pair) == original_hashes
    if args.run:
        summary = summarize(final)
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "checksums.json", {
            str(path.relative_to(args.output)):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output.rglob("*")) if path.is_file()
        })
        print(json.dumps(summary, indent=2))
    else:
        print("Preview passed. No training started and no results written.")


if __name__ == "__main__":
    main()
