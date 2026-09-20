# Experiment data record

Organized on 2026-09-20. Raw CSV contents are preserved unchanged.

New controlled experiments are stored separately under `experiments/`. The next test is [three matched seeds at alpha 0.1 and scale 10](experiments/2026-09-20_alpha_0.1_scale_10_three_seeds/REPORT.md). Each experiment has its own configurations, source snapshots, checksums, and summary; the historical catalog remains unchanged.

## Where to find data

| Location | Meaning | Permitted use |
| --- | --- | --- |
| `baselines/initial_alpha_sweep/` | Five recent no-attack runs | Preliminary baseline plots; not three controlled repeats |
| `archive/legacy/` | Older CSVs, plots, and source snapshots, grouped as before | Historical reference; consult catalog status before analysis |
| `archive/recovered_index/` | CSVs missing from disk but preserved in the pre-cleanup Git index | Provenance recovery only; duplicates are not extra runs |
| `catalog.csv` | Original/current paths, SHA-256 checksums, validity labels, final metrics | Inventory and traceability |

The recent baseline classification is inferred from the session and experiment runner, not independently verified per-run metadata. The alpha values come from filenames. The current `config.json` is not a record of every previous run.

## Preliminary baseline results

| Alpha | Round-10 accuracy |
| --- | --- |
| 0.01 | 65.81% |
| 0.1 | 87.27% |
| 1 | 90.72% |
| 10 | 90.95% |
| 100 | 90.82% |

Each row is one run. Do not calculate repeat statistics by treating rounds as independent runs.

## Why the old poisoning results need reruns

The CPU parameter exporter returned NumPy views sharing the model's memory. Training modified the saved pre-training arrays, making the calculated model update zero. The active `client.py` now copies these arrays. Regression tests exercise actual training and scales 1, 10, and 100.

The archived attack scripts retain the original bug for historical traceability. Files labeled `exclude_from_attack_analysis` cannot support conclusions about attack strength or malicious-client count. Their values have not been corrected or relabeled as verified clean runs. A baseline-named CSV inside an attack collection also requires provenance checks before reuse.

`unverified_legacy` and `unverified_recovered` indicate incomplete run metadata. Identical SHA-256 values identify byte-identical files; they must not be counted as independent experiments. Old plots inherit the limitations of their source data.

## Three-repeat protocol for the next experiments

1. Use seeds 42, 43, and 44 for each setting, controlling NumPy and PyTorch. `run_matched.py` implements this protocol; the older `run_all.py` does not.
2. Within each seed, match clean and attacked runs on initial server weights, partition, client training randomness, and training settings.
3. Save each run in a unique directory with its exact configuration, source commit, dependency versions, client sample counts, and per-round CSV. Do not overwrite this archive.
4. Report the mean and sample standard deviation of round-10 accuracy across the three runs, and the paired clean-minus-attacked accuracy drop.

Three repeats assess variability; they do not establish implementation correctness. No new training runs were performed during the initial archive cleanup; subsequent experiments live under `experiments/`. The older runner still writes files in the repository root and can overwrite results on repeated invocations. The new runner requires a fresh output directory.

Run `python3 scripts/verify_result_catalog.py` to check the archived inventory.
