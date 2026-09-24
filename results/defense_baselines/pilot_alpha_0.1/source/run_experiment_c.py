"""Experiment C: plan, check partitions, or train matched equal-size clients."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from source_compatibility import same_python_structure

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fl-experiment-c-mpl")

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]


# 讀取 CSV 並保留欄位名稱；數字在需要計算時才轉成 float。
def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# 只檢查 MNIST 的資料分配並畫圖，不訓練模型。
# 回傳每個 Alpha／seed 的樣本數、類別數量、分布差異與分配雜湊。
def check_partitions(alphas, output):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.datasets import MNIST

    from balanced_partition import balanced_partition

    # 只使用已下載的資料；預檢不會啟動訓練，也不需要連線下載。
    data = MNIST(
        str(ROOT / "data"),
        train=True,
        download=False,
    )

    output.mkdir(parents=True, exist_ok=True)

    records = []

    for alpha in alphas:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 3.5),
            layout="constrained",
        )

        for seed, ax in zip(SEEDS, axes):
            parts = balanced_partition(
                data.targets,
                5,
                alpha,
                seed,
            )

            assert [len(p) for p in parts] == [12000] * 5

            assert np.array_equal(
                np.sort(np.concatenate(parts)),
                np.arange(60000),
            )

            # counts[client, digit]：該 client 擁有多少張該數字圖片。
            counts = np.array(
                [
                    np.bincount(
                        np.asarray(data.targets)[p],
                        minlength=10,
                    )
                    for p in parts
                ]
            )

            # Mean total variation distance from the global label distribution.
            # TV 比較各端數字比例與全體比例，越高代表分布差異越大；它不是攻擊損害。
            global_p = counts.sum(axis=0) / 60000

            tv = float(np.abs(counts / 12000 - global_p).sum(axis=1).mean() / 2)

            records.append(
                dict(
                    alpha=alpha,
                    seed=seed,
                    sample_counts=[len(p) for p in parts],
                    client_label_counts=counts.tolist(),
                    mean_label_tv=tv,
                    partition_sha256=[
                        hashlib.sha256(
                            np.asarray(
                                p,
                                dtype=np.int64,
                            ).tobytes()
                        ).hexdigest()
                        for p in parts
                    ],
                )
            )

            im = ax.imshow(
                counts / 12000,
                vmin=0,
                vmax=1,
                aspect="auto",
                cmap="Blues",
            )

            ax.set(
                title=f"Seed {seed}; mean TV={tv:.3f}",
                xlabel="Digit label",
                ylabel="Client",
            )

            ax.set_xticks(range(10))
            ax.set_yticks(range(5))

        fig.colorbar(
            im,
            ax=axes,
            label="Fraction within client",
        )

        fig.suptitle(f"Experiment C: alpha={alpha:g}; " "12,000 samples per client")

        fig.savefig(
            output / f"alpha_{alpha:g}_label_distribution.png",
            dpi=160,
        )

        plt.close(fig)

    (output / "partitions.json").write_text(json.dumps(records, indent=2) + "\n")

    for r in records:
        print(
            f"alpha={r['alpha']:g}, "
            f"seed={r['seed']}: "
            "5 x 12000 verified; "
            f"label TV={r['mean_label_tv']:.3f}"
        )

    return records


# 沿用已完成設定前的檢查入口。
# 確認原始檔、程式、每端 12,000 張、配對起點及攻擊倍率，再回傳各 seed 損害。
def validate_run(folder, alpha):
    import math
    import numpy as np

    checksums = json.loads((folder / "checksums.json").read_text())

    # 先確認保存的原始檔案沒有被修改，再決定能否沿用結果。
    for name, digest in checksums.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name

    # AST 比較只忽略排版與註解；不忽略任何訓練常數或運算差異。
    for name in [
        "client.py",
        "run_balanced.py",
        "balanced_partition.py",
    ]:
        assert same_python_structure(folder / "source" / name, ROOT / name), (
            "Training source changed; " "use a fresh study directory"
        )

    results = []

    for seed in SEEDS:
        configs = []
        ends = {}

        for mode in ["clean", "poisoned"]:
            run = folder / f"seed_{seed}" / mode

            cfg = json.loads((run / "config.json").read_text())
            configs.append(cfg)

            # 這裡是實驗 C 的固定條件，避免誤用 A/B 或其他輪數的輸出。
            expected = dict(
                seed=seed,
                alpha=alpha,
                poison_scale=10,
                enable_poison=mode == "poisoned",
                poison_client=0,
                rounds=10,
                num_clients=5,
                batch_size=32,
                local_epochs=1,
                optimizer="SGD",
                learning_rate=0.01,
                partition_method=("capacity_constrained_dirichlet_v1"),
                samples_per_client=12000,
            )

            assert all(
                cfg[k] == v for k, v in expected.items()
            ), "Wrong experiment settings"

            assert cfg["client_sample_counts"] == [12000] * 5

            with np.load(run / "partitions.npz") as archive:
                parts = [archive[f"client_{i}"] for i in range(5)]

                assert [len(p) for p in parts] == [12000] * 5

                assert np.array_equal(
                    np.sort(np.concatenate(parts)),
                    np.arange(60000),
                )

                assert [
                    hashlib.sha256(p.astype(np.int64).tobytes()).hexdigest()
                    for p in parts
                ] == cfg["partition_sha256"]

            rows = read_rows(run / "results.csv")

            assert [int(r["round"]) for r in rows] == list(range(1, 11))

            assert all(
                math.isfinite(float(r["loss"])) and 0 <= float(r["accuracy"]) <= 1
                for r in rows
            )

            updates = read_rows(run / "updates.csv")

            assert {
                (
                    int(r["round"]),
                    int(r["client"]),
                )
                for r in updates
            } == {(r, c) for r in range(1, 11) for c in range(5)}

            assert len(updates) == 50

            for r in updates:
                multiplier = 10 if (mode == "poisoned" and int(r["client"]) == 0) else 1

                assert math.isclose(
                    float(r["sent_update_norm"]),
                    multiplier * float(r["update_norm"]),
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                )

            ends[mode] = rows[-1]

        for field in [
            "initial_parameters_sha256",
            "partition_sha256",
            "shuffle_seeds",
        ]:
            assert configs[0][field] == configs[1][field]

        clean = ends["clean"]
        poisoned = ends["poisoned"]

        results.append(
            dict(
                alpha=alpha,
                seed=seed,
                clean_accuracy=float(clean["accuracy"]),
                poisoned_accuracy=float(poisoned["accuracy"]),
                clean_loss=float(clean["loss"]),
                poisoned_loss=float(poisoned["loss"]),
                accuracy_damage_pp=100
                * (float(clean["accuracy"]) - float(poisoned["accuracy"])),
                loss_damage=float(poisoned["loss"]) - float(clean["loss"]),
            )
        )

    return results


# 把同一階段的個別 seed 結果整理成 CSV、損害圖與簡短報告。
# accuracy 使用 0～1；accuracy_damage_pp 使用百分點，兩者單位不同。
def summarize(output, rows):
    summary = []

    for alpha in sorted({r["alpha"] for r in rows}):
        group = [r for r in rows if r["alpha"] == alpha]

        item = {
            "alpha": alpha,
            "scale": 10,
            "samples_per_client": 12000,
            "seeds": len(group),
        }

        for metric in [
            "clean_accuracy",
            "poisoned_accuracy",
            "clean_loss",
            "poisoned_loss",
            "accuracy_damage_pp",
            "loss_damage",
        ]:
            item[metric + "_mean"] = statistics.mean(r[metric] for r in group)

            item[metric + "_sample_std"] = statistics.stdev(r[metric] for r in group)

        summary.append(item)

    for name, data in [
        ("per_seed.csv", rows),
        ("summary.csv", summary),
    ]:
        with (output / name).open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(data[0]),
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(data)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        layout="constrained",
    )

    for ax, metric, label in zip(
        axes,
        [
            "accuracy_damage_pp",
            "loss_damage",
        ],
        [
            "Accuracy damage (percentage points)",
            "Loss increase",
        ],
    ):
        ax.errorbar(
            range(len(summary)),
            [r[metric + "_mean"] for r in summary],
            yerr=[r[metric + "_sample_std"] for r in summary],
            fmt="o-",
            capsize=4,
        )

        ax.set_xticks(
            range(len(summary)),
            [f"{r['alpha']:g}" for r in summary],
        )

        ax.set(
            xlabel="Alpha (settings equally spaced)",
            ylabel=label,
        )

        ax.axhline(
            0,
            color="gray",
            ls="--",
        )

        ax.grid(alpha=0.2)

    fig.suptitle(
        "Experiment C: equal sample counts; " "mean +/- sample SD over 3 seeds"
    )

    fig.savefig(
        output / "damage.png",
        dpi=160,
    )

    plt.close(fig)

    (output / "REPORT.md").write_text(
        "# 實驗 C：固定每端資料量\n\n"
        "每端 12,000 張，FedAvg 權重 20%；"
        "Scale = 10；seeds 42、43、44。\n\n"
        "![損害](damage.png)\n\n"
        "`summary.csv` 為平均與樣本標準差；"
        "`per_seed.csv` 為各 seed 結果。"
        "Accuracy 損害是 Clean 減 Poisoned（百分點）；"
        "Loss 損害是 Poisoned 減 Clean。\n\n"
        "此分配方法先抽取每類數字的 client 偏好，"
        "再依容量分配全部圖片。"
        "不是原版無容量限制的 Dirichlet 方法；"
        "應配合 preflight 的數字分布圖判斷 Alpha "
        "實際帶來的差異。"
        "A/C 的差異同時包含分配演算法變更，"
        "不能把所有差異直接歸因於權重。\n",
        encoding="utf-8",
    )


# C 的操作入口：預設預覽；--check 只檢查資料；--run 才訓練。
# pilot 先做兩個端點，full 補齊五個 Alpha；完成設定可驗證後沿用。
def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--phase",
        choices=[
            "pilot",
            "full",
        ],
        default="pilot",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "results" / "studies" / "equal_samples"),
    )

    action = parser.add_mutually_exclusive_group()

    action.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check real MNIST partitions "
            "and draw label distributions, "
            "without training"
        ),
    )

    action.add_argument(
        "--run",
        action="store_true",
        help=("Run full training for " "pending settings"),
    )

    args = parser.parse_args()

    alphas = (
        [0.01, 100]
        if args.phase == "pilot"
        else [
            0.01,
            0.1,
            1,
            10,
            100,
        ]
    )

    print(
        f"Experiment C ({args.phase}): "
        f"alphas={alphas}; "
        f"scale=10; "
        f"seeds={SEEDS}; "
        "clean + poisoned"
    )

    folders = [
        (
            alpha,
            args.output / f"alpha_{alpha:g}_scale_10",
        )
        for alpha in alphas
    ]

    for alpha, folder in folders:
        status = "validate existing" if folder.exists() else "6 training runs pending"

        print(f"{folder}: {status}")

    if not args.run and not args.check:
        print(
            "Plan only. "
            "--check checks partitions; "
            "--run starts training. "
            "No old model checkpoints are resumed."
        )
        return

    # 先檢查所有已有資料夾；半成品會在新訓練開始前被發現。
    if args.run:
        # Refuse incomplete or incompatible output
        # before doing any new training.
        for alpha, folder in folders:
            if folder.exists():
                validate_run(
                    folder,
                    alpha,
                )

    # --check 到這裡就會結束；--run 也先做同樣的分配檢查。
    check_partitions(
        alphas,
        args.output / "preflight" / args.phase,
    )

    if args.check:
        return

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 依設定順序收集結果，已完成且通過驗證的設定不重跑。
    rows = []

    for alpha, folder in folders:
        if not folder.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "run_balanced.py",
                    "--alpha",
                    str(alpha),
                    "--scale",
                    "10",
                    "--seeds",
                    "42",
                    "43",
                    "44",
                    "--rounds",
                    "10",
                    "--output",
                    str(folder),
                ],
                cwd=ROOT,
                check=True,
            )

        rows.extend(
            validate_run(
                folder,
                alpha,
            )
        )

    # Separate pilot/full summaries so neither
    # hides a previously completed stage.
    destination = args.output / args.phase

    destination.mkdir(exist_ok=True)

    summarize(
        destination,
        rows,
    )

    print(f"Complete: " f"{destination / 'summary.csv'}")


if __name__ == "__main__":
    main()
