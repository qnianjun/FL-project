# Backdoor alpha sweep

Run from the repository root using the existing virtual environment:

```bash
.venv/bin/python run_backdoor_sweep.py
```

This read-only preview validates the completed alpha=0.1 pilot at
`results/Backdoor_test/mnist_alpha_0.1_ratio_0.2`, regenerates all 15
alpha/seed partitions, and checks matched model initialization and shuffling.
It does not train, download MNIST, or write results.

Start training yourself with:

```bash
.venv/bin/python run_backdoor_sweep.py --run
```

Only alpha varies: 0.01, 0.1, 1, 10, 100. Every setting uses MNIST,
five equal-size clients with 12,000 samples each, client 0 as attacker,
20% attacker-local poisoning (2,400 fixed selected samples, including original
target labels), the white 3x3 trigger at rows/columns 25–27, and target label 0.
Seeds are 42, 43, 44; each condition runs 10 rounds, one local epoch per round,
batch size 32, and SGD learning rate 0.01. Clean and Backdoor conditions share
partitions, initial weights, and shuffle seeds. ASR excludes original target-label
test images. The sweep calls the unchanged pilot runner for new settings.

The pilot is reused only after exact source and environment compatibility,
complete artifact inventories and SHA-256 verification, dataset hash checks,
regenerated partition and per-condition configuration checks, all ten metric
rows per run, and recomputation of its summary from raw metrics. An absent,
incomplete, or incompatible pilot stops execution; it is never silently retrained.
Reuse leaves 24 new condition runs (240 federated rounds).

The default destination is `results/Backdoor_test/alpha_sweep`. It must be fresh,
even for preview. Any existing output directory stops execution, including a
completed sweep. Interrupted runs are preserved and never resumed or overwritten;
inspect them and select a fresh `--output` path to retry. `--pilot` can select a
compatible complete pilot elsewhere. No existing results are modified.

After successful training, the destination contains:

- `alpha_*/`: complete raw outputs and original per-setting provenance. The
  alpha=0.1 pilot is copied byte-for-byte, retaining its source and checksums.
- `per_seed.csv`: 15 final-round rows, with clean accuracy and ASR for both
  conditions and the two within-seed differences.
- `summary.csv`: five rows, with mean and sample SD (`n - 1`, three seeds) of
  each metric and each paired difference.
- `source/`, `provenance.json`, and `checksums.json`: sweep driver/test snapshots,
  pilot origin and checksum, settings, and hashes of the complete sweep.

Both differences are **Backdoor minus clean**, in percentage points. A negative
clean-accuracy change means accuracy fell; positive ASR increase means ASR rose.
Differences are computed within each seed before computing mean and sample SD;
they are not clipped. CSV reports are only produced for a completed sweep, so a
preview does not manufacture results for untrained settings.

Quick checks (synthetic data; no training):

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_backdoor*.py' -v
```
