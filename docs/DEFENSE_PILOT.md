# Defense-baseline pilot

Preview from the repository root:

```bash
.venv/bin/python run_defense_pilot.py
```

Start training yourself only when ready:

```bash
.venv/bin/python run_defense_pilot.py --run
```

Preview validates saved FedAvg baselines and matched configurations. It does
not train, download MNIST, create output directories or write result files.
The default destination is `results/defense_baselines/pilot_alpha_0.1/`.
Any existing destination, including an incomplete run, stops execution.
Use `--output` with a fresh directory for a separate attempt. Incompatible or
incomplete FedAvg sources stop the pilot; they are never silently retrained.

## Matrix

Each benchmark compares `fedavg`, `clip_fedavg`, and `coordinate_median`, using
seeds 42/43/44 and matched clean/attack conditions. All methods receive the
same initial model, partitions, poison selection and shuffle seeds. Defenses
are applied to clean conditions too, so their clean utility cost is measured.

| Benchmark | Attack | Logical clients | FedAvg source |
| --- | --- | --- | --- |
| Model-update poisoning | Client 0, update scale 10 | 5 | `results/studies/equal_samples/alpha_0.1_scale_10/` |
| Backdoor | Client 0, poison ratio 0.2 | 5 | `results/Backdoor_test/mnist_alpha_0.1_ratio_0.2/` |
| Sybil-Backdoor | Three identical attacker identities, poison ratio 0.2 | 7 | `results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/sybil_3/` |

Fixed: MNIST; alpha 0.1; capacity-constrained original five partitions of 12,000
samples; existing 784→128→ReLU→10 model; ten rounds; one local epoch; batch 32;
SGD learning rate 0.01; sequential single-thread deterministic CPU execution.
All identities participate every round and report 12,000 samples.

Backdoor uses the existing 3×3 white trigger (value 1.0, zero-based rows and
columns 25–27), target 0, and the same 2,400 selected attacker samples every
round. Selection includes original target images and uses seed+2000.
Sybil identities reuse exactly the original client-0 partition and poison
positions, with identical seed+1000 shuffle streams. Honest partition i uses
seed+1000+i. Unique attacker data stays 12,000; the three-identity coalition
has 3/7 of the submitted sample weight, versus 1/5 in the other benchmarks.
Clean-Sybil controls retain all seven identities without poisoning.

There are **54 benchmark/method/seed/condition entries**. Reusing 18 FedAvg
entries leaves **36 new runs (360 rounds)**. Clean runs for the two new methods
are retained separately per benchmark, even where their settings coincide;
they are not extra independent seeds.

## Exact defense definitions

Let global parameters be w, submitted client parameters wᵢ, and δᵢ = wᵢ − w.
Model-update poisoning is applied before this server step. The aggregation
function receives only w, submitted parameters, sample counts, and method;
it cannot use attacker labels or evaluation results.

1. **FedAvg:** unchanged Flower sample-count-weighted mean of submitted models.
2. **Clip then FedAvg:** compute one full-model L2 norm per submitted δᵢ,
   across all tensors including biases, in float64. Each round set
   τ = median({‖δᵢ‖₂}) across all logical identities, without sample weighting.
   Use δ′ᵢ = δᵢ × min(1, τ/‖δᵢ‖₂), taking factor 1 for a zero update.
   A zero threshold suppresses every nonzero update. Apply Flower
   sample-weighted FedAvg to w + δ′ᵢ. This scales magnitude only and preserves
   direction before native-dtype rounding. There is no tuned multiplier,
   historical smoothing, attacker filtering, or clean-data calibration.
3. **Coordinate-wise median:** compute each coordinate's unweighted median
   across δᵢ, then add w. Five or seven identities give an odd-sized median;
   each identity is one vote, including replicas. No clipping or subsequent
   FedAvg is applied.

Outputs preserve each reference tensor's shape and dtype. Norm calculations
and defense arithmetic use float64; reconstructed model arrays use the
original dtype before aggregation/loading. Mathematical clipped norms are
checked against τ with a numerical tolerance; logs also record the actual
reconstructed norms so native-dtype roundoff is visible. NaN/Inf or
incompatible arrays stop execution rather than silently dropping clients.

These are baselines, not proven protections. A median-derived threshold can
change with adversarial submissions. Median aggregation and clipping may
also reduce clean utility under heterogeneous data; neither guarantees
protection against duplicated identities or low-norm backdoor updates.

## Reuse and provenance

Reuse requires complete artifact inventories/checksums, compatible execution
versions, datasets, settings, regenerated partitions, initialization hashes,
shuffle seeds, poison selection, and raw metrics matching saved summaries.
The Backdoor/Sybil source checks require identical bytes. Experiment C's
saved partitioner/runner has formatting differences: reuse requires an
identical Python AST (including constants, docstrings and operation order),
plus all exact run/data checks. Both old/current hashes and the matching
rule are recorded; source drift that changes computation is rejected.

FedAvg condition files are copied unchanged into each condition's `original/`
subdirectory, with `reuse.json` identifying their source and hashes. They
retain their historical column names and units. Baseline source snapshots
and original manifests are preserved under `baseline_provenance/`; those
manifests describe the original result directories, not that snapshot folder.
No attack result is modified. New results save:

- Root configuration, environment/dependency versions, dataset hashes,
  provenance, and all required current source/test snapshots.
- Per-condition configuration with initial/partition hashes, logical mapping,
  sample counts, shuffling and poison positions; new-run `partitions.npz`.
- New `results.csv` with ten rounds and `aggregation.csv` with each submitted
  norm, threshold, clipping factor, clipped norm and reconstructed norm.
- `per_seed.csv` (27 paired rows), `summary.csv` (nine aggregate rows), and
  a final root `checksums.json`. Summary files are produced only on completion.

## Metrics and signs

Normal accuracy/loss always use the unmodified 10,000-image test set.
Backdoor/Sybil ASR uses a separate triggered view of 9,020 originally nonzero
images, evaluated for both clean and attack models. ASR is blank/null for the
model-update-poisoning benchmark, not fabricated as zero.

- Accuracy damage = clean accuracy − attack accuracy, in percentage points.
- Loss damage = attack loss − clean loss.
- Paired ASR increase = attack ASR − clean ASR, in percentage points.
- Defense clean-accuracy degradation = same-benchmark/same-seed FedAvg clean
  accuracy − defense clean accuracy, in percentage points.
- Defense clean-loss increase = defense clean loss − FedAvg clean loss.

Positive degradation means worse utility; negative values remain unmodified.
Use each method's own clean run for attack damage, and the corresponding
FedAvg clean run for defense cost. Means and sample SDs use the three seed
pairs (n−1 denominator); rounds are not independent samples. No significance
claim or defense-performance result is available until training is run.

Quick checks:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_defense_pilot.py' -v
```
