"""Validate a completed matched experiment and generate its report and figure."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fl-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('experiment', type=Path)
args = parser.parse_args()
root = args.experiment
for name, digest in json.loads((root / 'checksums.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
summary = json.loads((root / 'summary.json').read_text())
seeds = sorted({r['seed'] for r in summary['runs']})
curves = {'clean': [], 'poisoned': []}
counts = []
rows = []
for seed in seeds:
    configs = []
    scores = []
    for mode in curves:
        folder = root / f'seed_{seed}' / mode
        cfg = json.loads((folder / 'config.json').read_text())
        configs.append(cfg)
        with (folder / 'results.csv').open() as handle:
            results = list(csv.DictReader(handle))
        assert [int(r['round']) for r in results] == list(range(1, cfg['rounds'] + 1))
        scores.append(float(results[-1]['accuracy']))
        curves[mode].append([float(r['accuracy']) * 100 for r in results])
        with (folder / 'updates.csv').open() as handle:
            updates = list(csv.DictReader(handle))
        assert len(updates) == cfg['rounds'] * cfg['num_clients']
        assert {(int(r['round']), int(r['client'])) for r in updates} == {
            (r, c) for r in range(1, cfg['rounds'] + 1) for c in range(cfg['num_clients'])
        }
        for row in updates:
            scale = cfg['poison_scale'] if mode == 'poisoned' and int(row['client']) == 0 else 1
            norm, sent = float(row['update_norm']), float(row['sent_update_norm'])
            assert math.isfinite(norm) and math.isfinite(sent)
            assert math.isclose(sent, scale * norm, rel_tol=1e-5, abs_tol=1e-7)
        assert float(updates[0]['update_norm']) > 0
        with np.load(folder / 'partitions.npz') as partition:
            parts = [partition[f'client_{i}'] for i in range(5)]
            assert [len(p) for p in parts] == cfg['client_sample_counts']
            assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(60000))
    for key in ['seed', 'alpha', 'rounds', 'initial_parameters_sha256', 'partition_sha256', 'shuffle_seeds', 'client_sample_counts']:
        assert configs[0][key] == configs[1][key], key
    counts.append(configs[0]['client_sample_counts'][0])
    rows.append(f'| {seed} | {scores[0] * 100:.2f}% | {scores[1] * 100:.2f}% | {(scores[0] - scores[1]) * 100:.2f} |')
clean = [r[-1] for r in curves['clean']]
poisoned = [r[-1] for r in curves['poisoned']]
drops = [a-b for a,b in zip(clean, poisoned)]
assert math.isclose(statistics.mean(clean)/100, summary['clean_mean_accuracy'])
assert math.isclose(statistics.mean(poisoned)/100, summary['poisoned_mean_accuracy'])
assert np.allclose(drops, summary['paired_drops_percentage_points'])
fig, ax = plt.subplots(figsize=(8, 5))
for mode, values in curves.items():
    values = np.array(values)
    x = np.arange(1, values.shape[1] + 1)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.plot(x, mean, marker='o', label=mode)
    ax.fill_between(x, mean-std, mean+std, alpha=0.18)
ax.set(xlabel='Federated round', ylabel='MNIST test accuracy (%)',
       title=f"Alpha {configs[0]['alpha']}, update scale {configs[1]['poison_scale']:g}; {len(seeds)} matched seeds")
ax.legend(title='Mean ± sample SD')
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(root / 'accuracy.png', dpi=160)
report = f'''# Matched poisoning experiment

Five clients, {configs[0]['rounds']} rounds, one local epoch, SGD learning rate 0.01, batch size 32, MNIST, alpha {configs[0]['alpha']}, client 0 update scale {configs[1]['poison_scale']:g}. Seeds: {', '.join(map(str, seeds))}.

| Seed | Clean final accuracy | Poisoned final accuracy | Drop (percentage points) |
| --- | --- | --- | --- |
''' + '\n'.join(rows) + f'''

- Clean mean ± sample SD: **{statistics.mean(clean):.2f}% ± {statistics.stdev(clean):.2f} percentage points**.
- Poisoned mean ± sample SD: **{statistics.mean(poisoned):.2f}% ± {statistics.stdev(poisoned):.2f} percentage points**.
- Paired accuracy drop: **{statistics.mean(drops):.2f} ± {statistics.stdev(drops):.2f} percentage points**.

![Accuracy curves with sample standard deviation](accuracy.png)

## Checks and limits

All {len(seeds) * 2} runs completed. Paired runs have identical initial weight and partition hashes and client shuffle seeds. Each partition contains all 60,000 training samples exactly once. All {len(seeds) * 2 * configs[0]['rounds'] * 5} client updates were checked for the expected scale; the malicious client's first update is nonzero in each attacked run. Raw output checksums were verified.

Client 0 has {counts} samples across seeds, respectively (FedAvg weights clients by sample count). This changes the attacker's aggregation weight and may help explain the varying impact, but these runs do not isolate that effect.

These are three paired repeats at one alpha and one attack scale, not evidence of a general heterogeneity trend or statistical significance. The attack amplifies an otherwise normal update; it does not inject labels, triggers, or Sybil identities. Positive drops mean lower accuracy under attack; negative drops mean higher accuracy.

Execution uses sequential CPU clients and Flower's sample-weighted aggregation in fixed client order. It evaluates the shared global test set once per round instead of repeating the identical evaluation on five clients. This controls initialization and ordering; it is not a network deployment benchmark. Source snapshots, dataset hashes, configuration, environment versions, partitions, and update norms are included.

Preflight found and fixed a partition rounding bug that could omit final samples of a class. Historical data remains unchanged; these runs use the corrected partitioner and controlled PyTorch seeds.

Reproduce into a new directory:

```bash
.venv/bin/python run_matched.py --output results/experiments/NEW_RUN
.venv/bin/python scripts/report_matched.py results/experiments/NEW_RUN
```
'''
(root / 'REPORT.md').write_text(report)
print(report)
