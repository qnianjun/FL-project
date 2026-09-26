# Experimental clipping and similarity grouping pilot

This pilot tests identical-replica Sybil-Backdoor only. It makes no universal
Sybil-detection or defense claim. The rule below is fixed before defense
training; accuracy and ASR are not used to select or tune it.

## Fixed rule

For each submitted model, form the full-model update `delta_i = w_i - w` in
float64. Reuse the validated `clip_fedavg` implementation to obtain
`tau = median_i ||delta_i||_2` and `f_i = min(1, tau / ||delta_i||_2)`;
zero updates have factor 1. Reconstruct clipped parameters in the original
model dtype, exactly as that implementation does.

Compute cosines from the original submitted directions (clipping preserves
nonzero directions). Join nonzero updates when **cosine >= 0.999999**. This is
a fixed tolerance of 0.000001 around identical directions, rather than a
boundary fitted to honest-client scores or defense outcomes. Zero updates have
undefined cosine and no edges, including when several zero updates occur.
If tau is zero, all contributions are zero even if original nonzero directions
form groups.

Use undirected connected components, with members sorted and groups ordered
by their smallest logical ID. This deliberately includes transitive chains:
not every pair in a component must meet the edge criterion.

For K components G, let c_i be the clipped contribution after native-dtype
parameter reconstruction. The mathematical aggregation rule is:

`w_next = w + (1/K) * sum_G [(1/|G|) * sum_(i in G) c_i]`.

Implementation averages the equivalent reconstructed parameters within each
group and then across groups in float64, casting the result to the model dtype.
Each group gets exactly one vote, regardless of its number of identities.
This pilot accepts equal positive reported counts only; it does not define a
new policy for unequal client sizes. Grouping and aggregation receive no
attacker labels. Independent clean clients can also be grouped if their
updates meet the criterion.

## Evidence and limitations

The completed diagnostic contains 180 Sybil–Sybil pairs with cosine in
[0.9999999999999998, 1.0] (all round to 1.0), while the maximum of 360
honest–honest pairs is 0.78734350751552. These are observations from the original
FedAvg trajectory, including both clean-Sybil and backdoor-Sybil conditions.
They do not guarantee separation after this defense changes the trajectory.
Correlated honest updates, transitive chains, nonidentical/adaptive Sybils,
and replica manipulation of the median clipping norm remain limitations.
Clean duplicated identities are structurally labelled replicas for analysis;
they are not malicious updates in the clean condition.

## Experiment and validation

MNIST; alpha 0.1; original five 12,000-sample partitions; mapping
`[0,1,2,3,4,0,0]`; poison ratio 0.2; target 0; white 3×3 trigger at rows/columns
25–27. Seeds 42,43,44; 10 rounds; one local epoch; batch 32; SGD lr 0.01.
Initialization, partition hashes, data hashes and shuffle seeds must match the
validated Sybil setup. Normal test accuracy/loss and ASR on the separate 9,020
triggered non-target test images are retained.

Six new runs compare matched clean-Sybil and backdoor-Sybil. Eighteen completed
condition runs for FedAvg, clip_fedavg and coordinate_median are reused only
with exact source/environment/configuration/partition compatibility and
validated raw metrics. Existing or incomplete output directories are rejected.
Default output: `results/defense_baselines/clip_similarity_group_pilot/`.
Configs, dependency versions, source snapshots, dataset/partition/initialization
hashes, provenance and completion checksums are retained. No full vectors are
saved.

`per_seed.csv` and `summary.csv` compare final clean accuracy/loss, ASR, paired
ASR increase, accuracy/loss damage and clean-model degradation versus FedAvg;
summary SD uses n−1 across three seeds. Per-run `aggregation.csv`,
`pairwise_similarity.csv`, `groups.csv` and `grouping_per_round.csv` preserve
all rounds. Grouping summaries first average within each seed across rounds,
then report the mean and sample SD across seeds, separately by condition.

Analysis-only effective replica coalition weight is
`sum_G (number of replica identities in G / |G|) / K`. It describes the linear
aggregation allocation after grouping and before individual clipping factors,
not causal influence on predictions. It is 20% if only the three replicas
merge and all four honest identities remain singletons. False grouping is
reported as honest identities in nonsingleton groups, honest–honest pairs
in the same group, and mixed replica/honest groups. Labels enter only after
aggregation has returned the model.

Preview (no training or result writes):

```bash
.venv/bin/python run_similarity_group.py
```

User-started training, six new runs / 60 rounds:

```bash
.venv/bin/python run_similarity_group.py --run
```
