# Controlled Sybil-backdoor stage

Research question: does duplicating the attacker's identity amplify the backdoor
while its underlying unique training data remains unchanged?

Run from the repository root. Preview validates the existing pilot, local MNIST,
and all nine sybil-count/seed designs without training, downloading, or writing
results:

```bash
.venv/bin/python run_sybil_backdoor.py
```

Start training yourself:

```bash
.venv/bin/python run_sybil_backdoor.py --run
```

The fixed protocol is MNIST, capacity-constrained Dirichlet alpha=0.1, the
original five partitions of 12,000 samples, seeds 42/43/44, ten rounds, one local
epoch, batch size 32, SGD learning rate 0.01, and sample-weighted FedAvg. The
validated pilot model, local training, trigger, ASR evaluation, and round loop
are reused directly. There is no model-update scaling.

| Attacker identities | Honest identities | Reported total samples | Coalition weight |
| --- | --- | --- | --- |
| 1 | 4 | 60,000 | 1/5 = 20% |
| 2 | 4 | 72,000 | 2/6 = 33.3333% |
| 3 | 4 | 84,000 | 3/7 = 42.8571% |

There are always 60,000 unique training samples and exactly 12,000 unique
attacker samples. The original identity order is retained and replicas are
appended: `[0, 1, 2, 3, 4]`, `[0, 1, 2, 3, 4, 0]`, or
`[0, 1, 2, 3, 4, 0, 0]`. Array positions are distinct logical identities; the
values identify their original partitions. Every identity reports 12,000.

All attacker identities use the same 2,400 selected local positions, selected
once using seed+2000, and independent shuffle generators with the same
seed+1000. This produces identical attacker updates, isolating multiplicity
rather than introducing additional shuffle randomness. Honest partition i uses
seed+1000+i. The clean control has the same duplication and shuffle streams,
without poisoned images or labels.

The poison ratio is 0.2 of all attacker samples, including original target
samples. The fixed white 3×3 trigger occupies zero-based rows/columns 25–27;
target label is 0. Clean accuracy/loss use the unchanged 10,000 normal test
images. ASR uses the separate triggered view of 9,020 originally nonzero images.

The alpha=0.1 pilot is reused only after exact source, dependency versions,
configuration, data, partition, initialization, shuffle, checksum, and metric
validation. Its six runs are copied without modification into `sybil_1/`.
Two additional counts × three seeds × two conditions require **12 new runs**
(120 federated rounds); there are 18 total condition runs including reuse.

Default output: `results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/`.
An existing output (including incomplete output) always stops execution. An
absent/incomplete/incompatible pilot also stops execution; there is no silent
retraining or resume. Use `--output` only with a fresh directory.

New results include configs, logical partitions and hashes, initialization
hashes, selected poison positions, sample counts, source snapshots, environment
and dataset provenance, raw per-round metrics, and checksums. `per_seed.csv`
contains nine final paired rows; `summary.csv` contains three means/sample-SDs
(n−1 denominator), including coalition weight. Differences are Backdoor minus
clean: accuracy/ASR differences are percentage points, loss differences are
loss units. Negative accuracy change means degradation. Small improvements
are retained, not clipped. Summary CSVs are written only after all runs validate.

Quick tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_sybil_backdoor.py' -v
```
