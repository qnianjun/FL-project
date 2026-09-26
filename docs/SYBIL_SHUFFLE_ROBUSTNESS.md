# Different-shuffle Sybil robustness test

This is one fixed robustness test, not a sweep or a universal robustness claim.
No defense threshold or shuffle seed is selected using accuracy or ASR.

Only appended replica identities 5 and 6 change their training-loader generator
seeds. Define `shuffle_seed = experiment_seed + 1000 + logical_identity`.
The same assignment is used in clean and backdoor runs and initialized once
per run; generator states advance normally across rounds.

| Experiment seed | Replica 0 | Replica 5 | Replica 6 |
|---|---:|---:|---:|
| 42 | 1042 | 1047 | 1048 |
| 43 | 1043 | 1048 | 1049 |
| 44 | 1044 | 1049 | 1050 |

Honest identities 1–4 keep `seed + 1000 + original partition ID`; replica 0
also keeps its original shuffle stream. Distinct seeds remove the guarantee
of identical replica updates; they do not guarantee that every update differs.

Everything else stays fixed: MNIST, alpha 0.1, five original 12,000-sample
partitions, logical mapping `[0,1,2,3,4,0,0]`, 12,000 unique attacker samples,
poison ratio 0.2, identical selected poison positions, target 0, white 3×3
trigger at rows/columns 25–27, seeds 42/43/44, 10 rounds, one local epoch,
batch 32, SGD learning rate 0.01, CPU deterministic execution. Clean/backdoor
pairs share initial parameters, partitions, and the entire shuffle assignment.

The defense implementation is unchanged: validated median-norm clipping,
cosine >= 0.999999, deterministic connected components, one vote per group.
Grouping/aggregation has no attacker labels. Labels are added afterward for
analysis. Zero-norm cosine remains undefined (blank CSV cell).

Six new runs / 60 rounds compare against 24 validated existing condition runs:
identical-replica FedAvg, clip_fedavg, coordinate_median, and
clip_similarity_group. Existing source, environment, data, partition,
configuration, raw metrics and archive checksums are validated before reuse.
No existing result directory may be overwritten or resumed.

The direct robustness comparison is between the two clip_similarity_group
settings, with shuffle assignment as the only changed experimental setting.
Comparisons to the other methods change both shuffle behavior and aggregation.
`defense_clean_accuracy_degradation_pp` retains the original definition:
original identical-replica FedAvg clean accuracy minus the current clean
accuracy. It therefore includes the shuffle change, not only defense cost.
No different-shuffle FedAvg control is run in this six-run experiment.

Output: `results/defense_baselines/sybil_distinct_shuffle/`.
The new-run folder uses the comparison label
`clip_similarity_group_distinct_shuffle`; its config still identifies the
actual unchanged method as `clip_similarity_group`.

Per run, `aggregation.csv` records submitted full-model L2 norms, median
thresholds, clipping factors and post-clipping norms; `groups.csv` records
members, sizes and weights; `pairwise_similarity.csv` records all 21 pairs
per round. `pairwise_similarity_labeled.csv` adds honest_honest,
sybil_honest and sybil_sybil labels only after aggregation.
`grouping_per_round.csv` records effective coalition weight and false grouping
of honest identities. Weight is the post-group allocation before individual
clipping factors, and clean replicas are structural labels, not malicious
updates. No full update vectors are saved.

`per_seed.csv` and `summary.csv` report final-round normal accuracy/loss, ASR,
paired ASR increase and clean-accuracy degradation for all five comparison
settings. Summary SD uses n−1 across the three seeds. Grouping diagnostics
are summarized first across rounds within each seed, then across seeds in
`grouping_per_seed.csv` and `grouping_summary.csv`. Configs, initialization and
partition hashes, datasets/dependency hashes and versions, source snapshots,
provenance and completion checksums are saved.

Preview only:

```bash
.venv/bin/python run_sybil_shuffle_robustness.py
```

User-started training:

```bash
.venv/bin/python run_sybil_shuffle_robustness.py --run
```
