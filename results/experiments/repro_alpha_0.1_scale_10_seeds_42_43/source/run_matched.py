"""Run three matched clean/poisoned MNIST experiments without network servers."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from flwr.server.strategy.aggregate import aggregate

import client


# 計算陣列內容的雜湊；用於比較配對條件，不把模型權重印到終端機。
def digest_arrays(arrays):
    return hashlib.sha256(b"".join(np.asarray(a).tobytes() for a in arrays)).hexdigest()


# 將設定、環境或摘要保存成縮排清楚的 JSON。
def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


# 執行 A/B 中的一個 Alpha／Scale 設定，依序完成多個 seeds 的配對訓練。
# 此程式是單一設定的執行器；整組實驗的安排由 run_study.py 處理。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.alpha <= 0 or args.scale <= 0 or args.rounds < 1:
        parser.error("alpha, scale, and rounds must be positive")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two distinct seeds")
    # 輸出目錄不可已存在，防止重跑時覆蓋舊資料。
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # 記錄程式快照與套件版本，讓結果能追溯到當時的程式。
    source = args.output / "source"
    source.mkdir()
    for name in ["client.py", "run_matched.py", "config.json"]:
        shutil.copy2(name, source / name)
    write_json(
        args.output / "environment.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.iterdir()
            },
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "numpy", "flwr"]
            },
            "execution": "Sequential CPU clients; Flower aggregate; all 5 clients every round; global test set evaluated once",
            "torch_threads": 1,
            "deterministic_algorithms": True,
        },
    )
    datasets_pair = client.load_data()
    write_json(
        args.output / "dataset_hashes.json",
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("data/MNIST/raw").iterdir())
            if p.is_file() and p.suffix != ".gz"
        },
    )
    client.ALPHA = args.alpha
    client.POISON_CLIENT = 0
    client.POISON_SCALE = args.scale
    final = []
    # 同一 seed 內配對比較，不把不同 seed 的資料混在一起訓練。
    for seed in args.seeds:
        paired_identity = None
        for attack in (False, True):
            mode = "poisoned" if attack else "clean"
            folder = args.output / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            client.SEED = seed
            client.ENABLE_POISON = attack
            workers = [client.FlowerClient(cid, datasets_pair) for cid in range(5)]
            initial = workers[0].get_parameters({})
            indices = [np.asarray(w.trainset.indices, dtype=np.int64) for w in workers]
            all_indices = np.concatenate(indices)
            assert np.array_equal(
                np.sort(all_indices), np.arange(len(datasets_pair[0]))
            ), "Partition must cover all training samples exactly once"
            identity = {
                "initial_parameters_sha256": digest_arrays(initial),
                "partition_sha256": [digest_arrays([i]) for i in indices],
            }
            if paired_identity is None:
                paired_identity = identity
            else:
                assert identity == paired_identity, "Clean/attack pair is not matched"
            np.savez_compressed(
                folder / "partitions.npz",
                **{f"client_{i}": value for i, value in enumerate(indices)},
            )
            write_json(
                folder / "config.json",
                {
                    "seed": seed,
                    "alpha": args.alpha,
                    "enable_poison": attack,
                    "poison_client": 0,
                    "poison_scale": args.scale,
                    "rounds": args.rounds,
                    "num_clients": 5,
                    "batch_size": 32,
                    "local_epochs": 1,
                    "optimizer": "SGD",
                    "learning_rate": 0.01,
                    "shuffle_seeds": [seed + 1000 + i for i in range(5)],
                    "client_sample_counts": [len(w.trainset) for w in workers],
                    "client_label_counts": [
                        np.bincount(
                            np.asarray(datasets_pair[0].targets)[i], minlength=10
                        ).tolist()
                        for i in indices
                    ],
                    **identity,
                },
            )
            # 每個條件重新使用初始模型，確保比較起點一致。
            parameters = initial
            with (folder / "results.csv").open("w", newline="") as result_file, (
                folder / "updates.csv"
            ).open("w", newline="") as update_file:
                result_writer = csv.writer(result_file, lineterminator="\n")
                update_writer = csv.writer(update_file, lineterminator="\n")
                result_writer.writerow(["round", "loss", "accuracy"])
                update_writer.writerow(
                    [
                        "round",
                        "client",
                        "update_norm",
                        "sent_update_norm",
                        "scale_ratio",
                    ]
                )
                for round_number in range(1, args.rounds + 1):
                    updates = []
                    for worker in workers:
                        weights, count, metrics = worker.fit(parameters, {})
                        norm = metrics["update_norm"]
                        sent = metrics["sent_update_norm"]
                        expected_scale = args.scale if attack and worker.cid == 0 else 1
                        assert math.isfinite(norm) and math.isfinite(sent)
                        assert math.isclose(
                            sent, expected_scale * norm, rel_tol=1e-5, abs_tol=1e-7
                        )
                        update_writer.writerow(
                            [
                                round_number,
                                worker.cid,
                                norm,
                                sent,
                                sent / norm if norm else "",
                            ]
                        )
                        updates.append((weights, count))
                    # FedAvg 以各 client 的樣本數加權，不是固定每端 20%。
                    parameters = aggregate(updates)
                    loss, _, metrics = workers[0].evaluate(parameters, {})
                    accuracy = metrics["accuracy"]
                    assert math.isfinite(loss) and 0 <= accuracy <= 1
                    result_writer.writerow([round_number, loss, accuracy])
                    result_file.flush()
                    update_file.flush()
                    print(
                        f"seed={seed} {mode} round={round_number}/{args.rounds} accuracy={accuracy:.4f} loss={loss:.4f}",
                        flush=True,
                    )
            final.append(
                {"seed": seed, "condition": mode, "accuracy": accuracy, "loss": loss}
            )
    clean = [r["accuracy"] for r in final if r["condition"] == "clean"]
    poisoned = [r["accuracy"] for r in final if r["condition"] == "poisoned"]
    # 準確率是 0～1，乘 100 後損害的單位才是「百分點」。
    drops = [(a - b) * 100 for a, b in zip(clean, poisoned)]
    summary = {
        "runs": final,
        "clean_mean_accuracy": statistics.mean(clean),
        "clean_sample_std": statistics.stdev(clean),
        "poisoned_mean_accuracy": statistics.mean(poisoned),
        "poisoned_sample_std": statistics.stdev(poisoned),
        "paired_drops_percentage_points": drops,
        "mean_drop_percentage_points": statistics.mean(drops),
        "drop_sample_std_percentage_points": statistics.stdev(drops),
    }
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "checksums.json",
        {
            str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
