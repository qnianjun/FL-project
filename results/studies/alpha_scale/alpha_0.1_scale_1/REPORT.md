# Matched poisoning experiment

Five clients, 10 rounds, one local epoch, SGD learning rate 0.01, batch size 32, MNIST, alpha 0.1, client 0 update scale 1. Seeds: 42, 43, 44.

| Seed | Clean final accuracy | Poisoned final accuracy | Drop (percentage points) |
| --- | --- | --- | --- |
| 42 | 87.50% | 87.50% | 0.00 |
| 43 | 85.33% | 85.33% | 0.00 |
| 44 | 75.77% | 75.77% | 0.00 |

- Clean mean ± sample SD: **82.87% ± 6.24 percentage points**.
- Poisoned mean ± sample SD: **82.87% ± 6.24 percentage points**.
- Paired accuracy drop: **0.00 ± 0.00 percentage points**.

Accuracy damage = clean accuracy minus poisoned accuracy (percentage points).
Loss damage = poisoned loss minus clean loss. Positive damage means worse performance.

- Clean final loss: **0.5274 ± 0.1222**.
- Poisoned final loss: **0.5274 ± 0.1222**.
- Paired loss damage: **0.0000 ± 0.0000**.

![Accuracy curves with sample standard deviation](accuracy.png)
![Loss curves with sample standard deviation](loss.png)

## Checks and limits

All 6 runs completed. Paired runs have identical initial weight and partition hashes and client shuffle seeds. Each partition contains all 60,000 training samples exactly once. All 300 client updates were checked for the expected scale; the malicious client's first update is nonzero in each attacked run. Raw output checksums were verified.

Client 0 has [14100, 1135, 17912] samples across seeds, respectively (FedAvg weights clients by sample count). This changes the attacker's aggregation weight and may help explain the varying impact, but these runs do not isolate that effect.

These are three paired repeats at one alpha and one attack scale, not evidence of a general heterogeneity trend or statistical significance. The attack amplifies an otherwise normal update; it does not inject labels, triggers, or Sybil identities. Positive drops mean lower accuracy under attack; negative drops mean higher accuracy.

Execution uses sequential CPU clients and Flower's sample-weighted aggregation in fixed client order. It evaluates the shared global test set once per round instead of repeating the identical evaluation on five clients. This controls initialization and ordering; it is not a network deployment benchmark. Source snapshots, dataset hashes, configuration, environment versions, partitions, and update norms are included.

Preflight found and fixed a partition rounding bug that could omit final samples of a class. Historical data remains unchanged; these runs use the corrected partitioner and controlled PyTorch seeds.

Reproduce into a new directory:

```bash
.venv/bin/python run_matched.py --alpha 0.1 --scale 1 --seeds 42 43 44 --rounds 10 --output results/experiments/NEW_RUN
.venv/bin/python scripts/report_matched.py results/experiments/NEW_RUN
```
