# Attack results summary

Scope: completed, saved MNIST experiments only. No training or raw-result changes were made to produce this document.

## Reading the results

- **Shared settings:** MNIST (60,000 training / 10,000 test images); 784→128→ReLU→10 model; seeds 42/43/44; 10 rounds; one local epoch; batch 32; SGD learning rate 0.01. Five original clients, client 0 malicious; all identities participate each round. Sequential deterministic CPU execution. Sybil adds logical identities; D changes aggregation.
- **Matching:** within each seed and setting, clean/attack conditions share initial parameters, partitions and shuffle seeds. Reported results are from round 10. “±” denotes sample SD across three seeds (denominator n−1), not a confidence interval. Tables without ± show means or explicitly named seeds.
- **Signs:** model-poisoning damage = clean accuracy − attacked accuracy, in percentage points (pp); positive is worse. Backdoor/Sybil Δaccuracy and ΔASR = attack − matched clean, in pp. Loss change always means attack − clean. Negative damage and positive accuracy changes are retained.
- **Evidence:** checked 738 manifest-listed files, 148 saved condition copies / 1,480 round rows, configs, source snapshots, partition identities, per-seed values and recomputed summaries. Copies are not independent experiments. No formal significance test is recorded in the inspected outputs; no statistical-significance claims are made here.
- **Consistency:** no numerical or checksum mismatch found. Earlier C/D status notes described unfinished work despite completed outputs; the [C README](../results/studies/equal_samples/README.md) and [results index](../results/INDEX.md) have now been updated during repository organization. D references A’s saved partitions through `environment.json`; it does not save separate partition files. These are provenance/documentation points, not missing D training results.

## 1. Model-update poisoning

This attack **scales a normally trained model update**, not images or labels:
`sent = starting_parameters + scale × (local_parameters − starting_parameters)`.
Main metrics are normal accuracy, cross-entropy loss, and within-seed accuracy/loss damage.

### A. Alpha sweep, scale = 10

**Question:** how does the attack outcome change across data distributions? **Fixed:** shared settings, scale 10, original unconstrained Dirichlet partitions, sample-weighted FedAvg. **Variable:** alpha = 0.01/0.1/1/10/100.

| Alpha | Clean acc. % | Attacked acc. % | Damage pp ± SD | Loss increase, mean |
| --- | --- | --- | --- | --- |
| 0.01 | 69.96 | 43.20 | 26.76 ± 27.87 | 13.12 |
| 0.1 | 82.87 | 48.65 | 34.22 ± 30.08 | 4.58 |
| 1 | 90.58 | 42.20 | 48.38 ± 24.11 | 6.65 |
| 10 | 90.92 | 28.21 | 62.71 ± 18.33 | 10.57 |
| 100 | 90.93 | 26.25 | 64.68 ± 4.45 | 7.88 |

**Measured:** mean accuracy damage rises from 26.76 to 64.68 pp, with large seed variation at smaller alpha. **Supported interpretation:** attack impact depends on the experimental setting. **Confounder:** alpha changes both label composition and client sample counts/FedAvg weights; this trend cannot be attributed solely to Non-IID. For example, at alpha 0.1 the attacker weights are 23.50%, 1.89%, and 29.85% for seeds 42/43/44.

Sources: [summary](../results/studies/alpha_scale/summary.csv), [per seed](../results/studies/alpha_scale/per_seed.csv), [study manifest and original run paths](../results/studies/alpha_scale/study.json).

### B. Scale sweep, alpha = 0.1

**Question:** how much damage results from stronger update amplification? **Fixed:** shared settings, alpha 0.1, the same original partitions within each seed, sample-weighted FedAvg. **Variable:** scale = 1/2/5/10/20; scale 1 is a no-amplification control.

| Scale | Clean acc. % | Attacked acc. % | Damage pp ± SD | Loss increase, mean |
| --- | --- | --- | --- | --- |
| 1 | 82.87 | 82.87 | 0.00 ± 0.00 | 0.00 |
| 2 | 82.87 | 81.58 | 1.28 ± 1.50 | 0.03 |
| 5 | 82.87 | 77.87 | 5.00 ± 4.45 | 0.14 |
| 10 | 82.87 | 48.65 | 34.22 ± 30.08 | 4.58 |
| 20 | 82.87 | 42.82 | 40.05 ± 37.95 | 22.37 |

**Measured:** mean damage increases with scale; scale 1 has zero accuracy damage and only ≈1.07×10⁻⁸ mean loss difference. **Supported interpretation:** update amplification can severely disrupt learning. **Limitation:** this is a mean trend, not a per-seed law: seed 44 damage falls from 46.38 pp at scale 10 to 43.52 pp at scale 20. Seed 43 remains comparatively insensitive under these weighted partitions.

Sources: the same [A/B summaries](../results/studies/alpha_scale/summary.csv) and [per-seed records](../results/studies/alpha_scale/per_seed.csv). A/B share alpha 0.1, scale 10: nine unique settings / 54 training runs, not ten independent settings.

### C. Equal-size clients, alpha sweep

**Question:** what happens when each client has the same number of images? **Fixed:** shared settings, scale 10, 12,000 images/client, sample-weighted FedAvg (20% each), capacity-constrained Dirichlet partitioning. **Variable:** alpha.

| Alpha | Clean acc. % | Attacked acc. % | Damage pp ± SD | Loss increase, mean |
| --- | --- | --- | --- | --- |
| 0.01 | 85.56 | 23.30 | 62.26 ± 5.70 | 6.44 |
| 0.1 | 87.70 | 28.03 | 59.67 ± 3.99 | 4.87 |
| 1 | 90.73 | 34.67 | 56.05 ± 8.46 | 4.97 |
| 10 | 90.89 | 31.38 | 59.51 ± 2.33 | 7.16 |
| 100 | 90.99 | 34.53 | 56.46 ± 4.37 | 5.58 |

**Measured:** large mean damage persists (56.05–62.26 pp), without A’s increasing alpha–damage pattern. **Supported interpretation:** equal sample counts do not remove vulnerability to scaled updates. **Confounder:** C changes the partition algorithm and realized data allocation as well as balancing counts. A versus C is not a pure one-variable causal comparison; the disappearance of A’s trend does not prove aggregation weight was its only cause.

Sources: [full summary](../results/studies/equal_samples/full/summary.csv), [per seed](../results/studies/equal_samples/full/per_seed.csv), [representative C config](../results/studies/equal_samples/alpha_0.1_scale_10/seed_43/poisoned/config.json).

### D. Sample-weighted versus equal-weight aggregation

**Question:** how does the aggregation rule change attack damage on the same data? **Fixed:** shared settings, alpha 0.1, scale 10, A’s exact saved partitions, initialization and shuffling. **Variable:** sample-weighted versus 20%-per-client averaging. This is an aggregation control, **not a defense**.

| Seed | Attacker images | Attacker weight: weighted → equal | Weighted damage pp | Equal damage pp | Equal − weighted damage pp |
| --- | --- | --- | --- | --- | --- |
| 42 | 14100 | 23.50% → 20% | 56.32 | 31.02 | -25.30 |
| 43 | 1135 | 1.89% → 20% | -0.04 | 26.31 | 26.35 |
| 44 | 17912 | 29.85% → 20% | 46.38 | 44.43 | -1.95 |

**Measured:** mean damage is 34.22 ± 30.08 pp weighted versus 33.92 ± 9.40 pp equal. Mean clean/attacked accuracy is 82.87%/48.65% weighted versus 84.37%/50.45% equal. Mean loss increase is 4.58 versus 2.13. Seed 43 changes from −0.04 to 26.31 pp damage.

**Supported interpretation:** aggregation weighting materially changes individual outcomes; D is the cleaner weighting control because it reuses the same partitions. **Limitation:** all five clients’ weights change, not only the attacker’s. Similar mean damage hides opposing seed effects; equal weighting is not consistently protective.

Sources: [comparison](../results/studies/weight_comparison/equal/comparison.csv), [per seed](../results/studies/weight_comparison/equal/per_seed.csv), [summary](../results/studies/weight_comparison/equal/summary.json), [source-partition provenance](../results/studies/weight_comparison/equal/environment.json).

### Seed-43 reproducibility audit

**Question:** was the near-zero weighted damage a one-off training failure? **Fixed:** original alpha 0.1, scale 10, weighted setup. **Variable:** a fresh execution of seeds 42 and 43, not new independent seeds. **Metrics:** full round accuracy/loss, update norms and final damage.

| Seed | Original = repeat clean acc. % | Original = repeat attacked acc. % | Original = repeat damage pp |
| --- | --- | --- | --- |
| 42 | 87.50 | 31.18 | 56.32 |
| 43 | 85.33 | 85.37 | −0.04 |

**Measured:** all ten metric rows and all update-log rows match exactly, with matching partitions, initial hashes and shuffle settings; saved Python source structures also match. The attacker’s update amplification is present in the logs. **Conclusion:** seed 43 was reproducibly near-zero damage under this original weighted setting, not a one-off failure. **Limitation:** replay establishes reproducibility, not immunity or the sole explanation; its attacker holds only 1,135/60,000 images (1.89% weight), and D shows sensitivity to weighting. Replays do not increase the number of independent seeds.

Sources: [audit summary](../results/experiments/repro_alpha_0.1_scale_10_seeds_42_43/summary.json), [audit environment](../results/experiments/repro_alpha_0.1_scale_10_seeds_42_43/environment.json), [original seed-43 config](../results/experiments/2026-09-20_alpha_0.1_scale_10_three_seeds/seed_43/poisoned/config.json).

## 2. Backdoor

**Fixed protocol:** shared settings; five equal-size capacity-constrained partitions; client 0; sample-weighted FedAvg (20% each); no update amplification. Poison 20% of the attacker’s data (2,400 fixed images, including original target-label images): white 3×3 bottom-right trigger, zero-based rows/columns 25–27, value 1.0, target 0. Normal accuracy/loss use the unchanged 10,000 test images. **ASR** is the percentage predicted as 0 among 9,020 triggered test images originally not labeled 0. Clean models receive the same ASR evaluation.

### Alpha = 0.1 pilot

**Question:** can a trigger-based attack raise ASR while normal accuracy stays close to its clean control? **Variable:** poisoning off/on; all protocol settings, including alpha 0.1, fixed.

| Seed | Normal acc. %: clean → attack | ASR %: clean → attack | Δaccuracy pp | ΔASR pp |
| --- | --- | --- | --- | --- |
| 42 | 89.20 → 89.07 | 0.81 → 53.78 | -0.13 | 52.97 |
| 43 | 88.86 → 88.79 | 1.00 → 67.04 | -0.07 | 66.04 |
| 44 | 85.03 → 84.97 | 0.59 → 68.29 | -0.06 | 67.71 |

**Measured:** mean normal accuracy 87.70% → 87.61%; ASR 0.80% → 63.04%; paired changes −0.09 ± 0.04 pp accuracy and +62.24 ± 8.07 pp ASR. Mean normal loss 0.4219 → 0.4299. **Conclusion:** the observed backdoor is substantial despite little normal-accuracy change. **Limitation:** one trigger, target, poison ratio and small model; normal accuracy alone cannot characterize attack success.

Sources: [pilot summary](../results/Backdoor_test/mnist_alpha_0.1_ratio_0.2/summary.json), [config](../results/Backdoor_test/mnist_alpha_0.1_ratio_0.2/config.json).

### Equal-size alpha sweep

**Question:** how does label-distribution heterogeneity relate to Backdoor outcomes at fixed client weight? **Fixed:** protocol above. **Variable:** alpha; matched clean/backdoor pair at each setting.

| Alpha | Normal acc. %: clean → attack | Clean ASR % | Attack ASR % ± SD | Δaccuracy pp ± SD | ΔASR pp ± SD |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 85.56 → 85.31 | 0.95 | 67.82 ± 28.08 | -0.25 ± 0.28 | 66.87 ± 27.82 |
| 0.1 | 87.70 → 87.61 | 0.80 | 63.04 ± 8.04 | -0.09 ± 0.04 | 62.24 ± 8.07 |
| 1 | 90.73 → 90.68 | 0.84 | 60.72 ± 5.48 | -0.05 ± 0.10 | 59.88 ± 5.47 |
| 10 | 90.89 → 90.86 | 0.82 | 60.69 ± 2.09 | -0.03 ± 0.06 | 59.87 ± 2.13 |
| 100 | 90.99 → 90.94 | 0.78 | 61.86 ± 0.56 | -0.05 ± 0.07 | 61.08 ± 0.60 |

**Measured:** no clear monotonic alpha-versus-mean-ASR trend. Low alpha has larger seed variability: at 0.01 the ASRs are 88.20%, 35.79%, 79.46% (SD 28.08 pp), versus SD 0.56 pp at 100. **Supported interpretation:** attack reliability varies even with fixed aggregation weight. **Limitation:** only three seeds; alpha changes realized label composition and which images are poisoned. No general causal law or significance is established. The alpha 0.1 pilot is reused byte-for-byte, not a new replicate.

Sources: [summary](../results/Backdoor_test/alpha_sweep/summary.csv), [per seed](../results/Backdoor_test/alpha_sweep/per_seed.csv), [reuse provenance](../results/Backdoor_test/alpha_sweep/provenance.json).

## 3. Sybil-Backdoor

**Question:** does duplicating the attacker’s identity amplify Backdoor without adding unique training data? **Fixed:** alpha 0.1 Backdoor protocol and original five partitions; attacker’s unique data remains client 0’s 12,000 images, honest partitions remain 1–4. Replicas use identical poison positions and shuffle streams. **Variable:** 1/2/3 logical attacker identities; each reports 12,000 samples. Server sees 4 + s identities, giving coalition weight s/(4+s). Clean-Sybil controls duplicate identities identically without poisoning. **Metrics:** normal accuracy/loss, ASR, matched changes and coalition weight.

| Sybil count / total identities | Coalition weight | Normal acc. %: clean → attack | Clean ASR % | Attack ASR % ± SD | Δaccuracy pp ± SD | ΔASR pp ± SD |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / 5 | 20.00% | 87.70 → 87.61 | 0.80 | 63.04 ± 8.04 | -0.09 ± 0.04 | 62.24 ± 8.07 |
| 2 / 6 | 33.33% | 86.61 → 86.77 | 0.86 | 93.23 ± 2.13 | 0.15 ± 0.23 | 92.37 ± 2.29 |
| 3 / 7 | 42.86% | 84.11 → 84.82 | 0.92 | 98.50 ± 0.64 | 0.72 ± 0.84 | 97.58 ± 1.18 |

**Measured:** ASR rises 63.04% → 93.23% → 98.50%, while clean-Sybil accuracy falls 87.70% → 86.61% → 84.11% (−3.59 pp from one to three identities). Attacked normal accuracy also falls across counts. Small positive *within-count* Δaccuracy at two/three identities does not erase that decline.

**Supported interpretation:** repeated identity contributions amplify the backdoor without additional unique data. **Limitation:** logical identity count and coalition weight increase together; this does not isolate an identity-count effect at fixed weight. Duplication also reweights the clean data distribution. Results use identical replicas, not varied or adaptive Sybils. Count 1 reuses the validated pilot; only counts 2/3 add training runs.

Sources: [summary](../results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/summary.csv), [per seed](../results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/per_seed.csv), [config](../results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/config.json), [reuse/preflight provenance](../results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/provenance.json).

## A. Cross-attack findings

- Model-update poisoning causes large normal-accuracy damage; Backdoor can have high ASR with almost unchanged normal accuracy. These measure different failure modes, not a shared “attack strength” score.
- D and Sybil provide evidence that contribution weighting matters. A’s alpha trend alone cannot separate label imbalance from weight changes; C does not make A-versus-C a causal comparison.
- Seed-specific outcomes matter: weighted seed 43 is reproducibly near-zero damage, and low-alpha Backdoor is variable. Reused runs and reruns are not extra independent evidence.

## B. What vulnerabilities appear common

**Interpretation, not an additional experiment:** the aggregation process admits harmful client contributions, and their influence depends on update magnitude and assigned weight. Duplicate identities can multiply one data source’s contribution. Ordinary clean testing can miss targeted trigger behavior. The experiments support these concerns in this MNIST setup, not universal claims about all FL systems.

## C. What a defense must address

**Requirements for future evaluation, not demonstrated defenses:** limit harmful update influence; account for identity duplication and coalition-level contribution; preserve useful learning from genuinely heterogeneous clients. Measure both clean accuracy/loss and triggered non-target ASR. Use matched no-attack controls, including clean-Sybil controls, and report per-seed outcomes plus mean/sample SD. Equal weighting is an experimental control, not an established defense. Any proposed defense must be tested across attack types and client distributions before claiming protection.

## D. Representative defense benchmarks

These are proposed evaluation choices based on the measured results, not completed defense tests. Retain all three seeds and each setting’s matched clean control.

| Benchmark | Settings to retain | Why include it |
| --- | --- | --- |
| Controlled model-update poisoning | C, alpha 0.1, scale 10, equal-size clients | Large damage (59.67 ± 3.99 pp) at fixed 20% attacker weight; shares the alpha 0.1 equal-size framework with Backdoor. |
| Update-amplification stress/control | B, alpha 0.1, scales 1/10/20 | No-amplification control, damaging attack and stronger stress case; retain original variable-size partitions. |
| Aggregation sensitivity | D, alpha 0.1, scale 10, both weighting rules | Same partitions; retain seed 43 as a reproducible low-damage weighted case, not an outlier to discard. |
| Backdoor base + variability checks | Alpha 0.1 primary; 0.01 and 100 secondary; poison ratio 0.2 | Shared pilot baseline, high seed variability and comparatively stable ASR. |
| Identity-amplification stress | Sybil counts 1/2/3, alpha 0.1, poison ratio 0.2 | Fixed unique attacker data; 20%/33.33%/42.86% coalition weights; includes clean-duplication utility cost. |
