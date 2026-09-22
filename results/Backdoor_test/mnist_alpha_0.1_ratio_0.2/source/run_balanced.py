"""Experiment C: matched runs with equal client sample counts; fresh training, not checkpoint continuation."""

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
from balanced_partition import balanced_partition
from torch.utils.data import DataLoader, Subset


class BalancedClient(client.FlowerClient):
    """Reuse the tested training/attack methods with a fixed-size data partition."""

    # 建立固定資料量的 client；沿用 FlowerClient 的訓練、攻擊、評估方法。
    # 模型初始化使用共同 seed，洗牌則為每個 client 使用獨立的 generator。
    def __init__(self, cid, datasets_pair, partitions, seed):
        self.cid = cid
        torch.manual_seed(seed)
        self.model = client.Net()
        self.trainset = Subset(datasets_pair[0], partitions[cid])
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 1000 + cid),
        )
        self.testset = datasets_pair[1]
        self.testloader = DataLoader(self.testset, batch_size=32, shuffle=False)


# 把陣列內容轉成雜湊值，用來確認 Clean／Poisoned 使用相同起點與分配。
def digest_arrays(arrays):
    return hashlib.sha256(b"".join(np.asarray(a).tobytes() for a in arrays)).hexdigest()


# 保存可閱讀的 JSON 設定或摘要，方便日後重現結果。
def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


# 執行實驗 C 的一個 Alpha／Scale 設定。
# 流程：讀取參數 → 保存來源 → 建立配對 → 訓練 → 統計 → 保存檢查碼。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not math.isfinite(args.alpha)
        or not math.isfinite(args.scale)
        or args.alpha <= 0
        or args.scale <= 0
        or args.rounds < 1
    ):
        parser.error("alpha, scale, and rounds must be positive")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two distinct seeds")
    # 必須是新資料夾，避免覆蓋結果；跨設定的沿用由 run_experiment_c.py 管理。
    args.output.mkdir(parents=True, exist_ok=False)
    # 固定 CPU 執行條件，降低不同執行順序帶來的差異。
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # 保存本次實際使用的原始碼，日後即使主程式改版仍能追溯。
    source = args.output / "source"
    source.mkdir()
    for name in [
        "client.py",
        "run_balanced.py",
        "balanced_partition.py",
        "config.json",
    ]:
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
    # 每個 seed 是一組重複實驗，內含 Clean 與 Poisoned 兩次訓練。
    for seed in args.seeds:
        # 每組配對只建立一次分配，兩個條件共用同一份圖片索引。
        partitions = balanced_partition(datasets_pair[0].targets, 5, args.alpha, seed)
        paired_identity = None
        for attack in (False, True):
            mode = "poisoned" if attack else "clean"
            folder = args.output / f"seed_{seed}" / mode
            folder.mkdir(parents=True)
            client.SEED = seed
            client.ENABLE_POISON = attack
            workers = [
                BalancedClient(cid, datasets_pair, partitions, seed) for cid in range(5)
            ]
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
                    "partition_method": "capacity_constrained_dirichlet_v1",
                    "samples_per_client": len(datasets_pair[0]) // 5,
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
            # 每個條件都從相同初始權重開始；不是接續前一個條件的訓練。
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
                    # 依樣本數合併各端模型；C 中各端樣本相同，因此權重皆為 20%。
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
    # 使用最後一輪的結果計算配對損害，保留負值，不強制改成零。
    clean = [r["accuracy"] for r in final if r["condition"] == "clean"]
    poisoned = [r["accuracy"] for r in final if r["condition"] == "poisoned"]
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
    # 最後保存檢查碼，作為本設定完成及原始輸出未被改寫的檢查依據。
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
