"""Plan or run the alpha and scale studies; training requires --run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from source_compatibility import same_python_structure

ROOT = Path(__file__).resolve().parent
ALPHAS = [0.01, 0.1, 1, 10, 100]
SCALES = [1, 2, 5, 10, 20]
SEEDS = [42, 43, 44]


# 建立 A/B 的設定表：A 只改 Alpha，B 只改 Scale。
def settings(experiment):
    rows = []
    if experiment in ("A", "all"):
        rows.extend(("A", alpha, 10) for alpha in ALPHAS)
    if experiment in ("B", "all"):
        rows.extend(("B", 0.1, scale) for scale in SCALES)
    return rows


# 建立設定的唯一名稱，讓 A/B 交會點只訓練一次。
def key(alpha, scale):
    return f"alpha_{alpha:g}_scale_{scale:g}"


# 沿用舊結果前，確認 seeds、訓練條件、程式結構、套件與資料集相符。
# 純註解與排版差異可接受，訓練參數或邏輯變更仍會被拒絕。
def check_configuration(folder, alpha, scale):
    summary = json.loads((folder / "summary.json").read_text())
    assert sorted((r["seed"], r["condition"]) for r in summary["runs"]) == sorted(
        (seed, mode) for seed in SEEDS for mode in ("clean", "poisoned")
    ), f"Wrong seeds or conditions: {folder}"
    for seed in SEEDS:
        for mode in ("clean", "poisoned"):
            cfg = json.loads(
                (folder / f"seed_{seed}" / mode / "config.json").read_text()
            )
            expected = dict(
                seed=seed,
                alpha=alpha,
                poison_scale=scale,
                enable_poison=mode == "poisoned",
                poison_client=0,
                rounds=10,
                num_clients=5,
                batch_size=32,
                local_epochs=1,
                optimizer="SGD",
                learning_rate=0.01,
            )
            assert all(
                cfg[k] == v for k, v in expected.items()
            ), f"Wrong settings: {folder}"
    # Reuse only outputs produced by the active training implementation.
    for name in ("client.py", "run_matched.py"):
        assert same_python_structure(
            folder / "source" / name, ROOT / name
        ), f"Source changed: {name}"
    env = json.loads((folder / "environment.json").read_text())
    import importlib.metadata
    import platform

    assert env["python"] == platform.python_version(), "Python version changed"
    assert all(
        importlib.metadata.version(name) == version
        for name, version in env["packages"].items()
    ), "Package versions changed"
    for name, digest in json.loads(
        (folder / "dataset_hashes.json").read_text()
    ).items():
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        ), "Dataset changed"


# 先計算每個 seed 的 Clean／Poisoned 差，再算三個配對差的平均與標準差。
# 不能把十輪訓練當成十個獨立樣本。
def write_summary(output, entries):
    # per_seed 保存每次配對；aggregate 保存每個設定的統計。
    per_seed, aggregate = [], []
    for entry in entries:
        folder = Path(entry["path"])
        values = []
        for seed in SEEDS:
            ends = {}
            for mode in ("clean", "poisoned"):
                with (folder / f"seed_{seed}" / mode / "results.csv").open() as handle:
                    ends[mode] = list(csv.DictReader(handle))[-1]
            clean_acc, poison_acc = (
                float(ends[m]["accuracy"]) for m in ("clean", "poisoned")
            )
            clean_loss, poison_loss = (
                float(ends[m]["loss"]) for m in ("clean", "poisoned")
            )
            row = dict(
                experiment=entry["experiment"],
                alpha=entry["alpha"],
                scale=entry["scale"],
                seed=seed,
                clean_accuracy=clean_acc,
                poisoned_accuracy=poison_acc,
                clean_loss=clean_loss,
                poisoned_loss=poison_loss,
                accuracy_damage_pp=100 * (clean_acc - poison_acc),
                loss_damage=poison_loss - clean_loss,
            )
            per_seed.append(row)
            values.append(row)
        combined = {k: entry[k] for k in ("experiment", "alpha", "scale")}
        for metric in (
            "clean_accuracy",
            "poisoned_accuracy",
            "clean_loss",
            "poisoned_loss",
            "accuracy_damage_pp",
            "loss_damage",
        ):
            combined[metric + "_mean"] = statistics.mean(r[metric] for r in values)
            combined[metric + "_sample_std"] = statistics.stdev(
                r[metric] for r in values
            )
        aggregate.append(combined)
    for name, rows in [("per_seed.csv", per_seed), ("summary.csv", aggregate)]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    (output / "README.md").write_text("""# Alpha and scale study results

See `study.json` for the exact source directory used by each setting, `per_seed.csv` for paired final-round measurements, and `summary.csv` for means and sample standard deviations across seeds 42, 43, and 44.

Accuracy damage (percentage points) = 100 × (clean accuracy − poisoned accuracy).
Loss damage = poisoned loss − clean loss.
Positive values mean the attack made performance worse; negative values mean improvement. Accuracy columns use fractions (0–1). Damage is paired within each seed before averaging.

Experiment A changes alpha with scale fixed at 10. Experiment B changes scale with alpha fixed at 0.1. Their shared alpha=0.1, scale=10 setting refers to the same runs and is not independent evidence. Repeated clean controls across scale settings are also not additional independent seeds. Scale 1 is a control: ordinary updates should give approximately zero damage.

Three seeds provide preliminary variability estimates, not proof of statistical significance. Alpha also changes client sample counts and aggregation weights; interpret this as the effect of the whole partitioning scheme, not label imbalance alone.
""")


# 安排 A/B 的完整流程。預設只顯示計畫，必須加 --run 才會開始訓練。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["A", "B", "all"], default="all")
    parser.add_argument(
        "--output", type=Path, default=Path("results/studies/alpha_scale")
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute pending full training runs; otherwise show the plan only",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse the two known completed experiment sets after validation",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    known = {
        key(0.01, 10): ROOT
        / "results/experiments/2026-09-20_alpha_0.01_scale_10_three_seeds",
        key(0.1, 10): ROOT
        / "results/experiments/2026-09-20_alpha_0.1_scale_10_three_seeds",
    }
    entries = []
    # 用唯一設定名稱去重；A/B 相同交會點共用一份原始結果。
    jobs = {}
    for experiment, alpha, scale in settings(args.experiment):
        setting = key(alpha, scale)
        folder = (
            known[setting]
            if args.reuse_existing and setting in known
            else output / setting
        )
        entries.append(
            dict(experiment=experiment, alpha=alpha, scale=scale, path=str(folder))
        )
        jobs[setting] = (alpha, scale, folder)
    pending = 0
    for setting, (_, _, folder) in jobs.items():
        complete = (folder / "checksums.json").is_file() and (
            folder / "summary.json"
        ).is_file()
        print(
            f'{setting}: {"validate/reuse" if complete else "run 6 trials"} -> {folder}'
        )
        pending += not complete
    print(
        f"{len(jobs)} unique settings; {pending * 6} training runs pending (completed outputs still require validation)."
    )
    # 安全預設：查看計畫不會啟動耗時的模型訓練。
    if not args.run:
        print(
            "Plan only. Add --run to execute. Existing incomplete directories are never overwritten."
        )
        return
    # Fail on incompatible or incomplete saved outputs before starting any training.
    for alpha, scale, folder in jobs.values():
        if folder.exists():
            check_configuration(folder, alpha, scale)
            subprocess.run(
                [sys.executable, "scripts/report_matched.py", str(folder)],
                cwd=ROOT,
                check=True,
            )
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(experiment=args.experiment, seeds=SEEDS, rounds=10, entries=entries)
    manifest_path = output / "study.json"
    if manifest_path.exists():
        assert (
            json.loads(manifest_path.read_text()) == manifest
        ), "Existing study has a different plan; use a new output directory"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for alpha, scale, folder in jobs.values():
        if not folder.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "run_matched.py",
                    "--alpha",
                    str(alpha),
                    "--scale",
                    str(scale),
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
        check_configuration(folder, alpha, scale)
        subprocess.run(
            [sys.executable, "scripts/report_matched.py", str(folder)],
            cwd=ROOT,
            check=True,
        )
    write_summary(output, entries)
    print(f'Study complete: {output / "summary.csv"}')


if __name__ == "__main__":
    main()
