# 歷史資料紀錄（2026-09-20 整理）

目前完整進度請看 [訓練結果索引](INDEX.md) 與 [攻擊結果摘要](../docs/ATTACK_RESULTS_SUMMARY.md)。A/B/C/D、重現檢查、Backdoor、Sybil-Backdoor 都已有完成結果。下方保留最初整理時的兩組實驗與歷史分類，不代表目前只完成這兩組。

Organized on 2026-09-20. Raw CSV contents are preserved unchanged.

## 當時可用的兩組結果（歷史子集）

At the time of the initial cleanup, the following two completed experiment sets were available. Both passed saved-result validation: checksums, paired initial weights and partitions, full training-data coverage, and update scaling. Each uses seeds 42, 43, and 44, five clients, ten rounds, and client 0 scaling its update by 10.

| Alpha | Clean accuracy, mean ± sample SD | Poisoned accuracy, mean ± sample SD | Paired drop, mean ± sample SD | Report |
| --- | --- | --- | --- | --- |
| 0.01 | 69.96% ± 6.30 pp | 43.20% ± 33.45 pp | 26.76 ± 27.87 pp | [Results and chart](experiments/2026-09-20_alpha_0.01_scale_10_three_seeds/REPORT.md) |
| 0.1 | 82.87% ± 6.24 pp | 48.65% ± 31.82 pp | 34.22 ± 30.08 pp | [Results and chart](experiments/2026-09-20_alpha_0.1_scale_10_three_seeds/REPORT.md) |

Here, pp means percentage points. There are **12 controlled runs total**, not 12 independent seeds: each setting has three paired repetitions. These results support describing attack effects at the tested settings; they do not yet establish that smaller alpha always causes more attack damage. Variation is large, and changing alpha also changes the attacking client's sample count and FedAvg weight.

The former `experiments/alpha_001/` folder was renamed to `experiments/2026-09-20_alpha_0.01_scale_10_three_seeds/`; its recorded configuration confirms alpha 0.01. Raw outputs were not changed. Each controlled experiment has its own configurations, source snapshots, checksums, and summary; the historical catalog remains unchanged.

## Where to find data

| Location | Meaning | Permitted use |
| --- | --- | --- |
| `experiments/2026-09-20_alpha_*/` | Controlled experiment folders; current selection is recorded in study.json | Follow the current INDEX.md and study manifest |
| `experiments/manual_unverified/` | Standalone result and configuration observed during cleanup | Preliminary observation only; not an additional controlled repeat |
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

## 原始三-seed 實驗規範

1. Use seeds 42, 43, and 44 for each setting, controlling NumPy and PyTorch. `run_matched.py` implements this protocol; the older `run_all.py` does not.
2. Within each seed, match clean and attacked runs on initial server weights, partition, client training randomness, and training settings.
3. Save each run in a unique directory with its exact configuration, source commit, dependency versions, client sample counts, and per-round CSV. Do not overwrite this archive.
4. Report the mean and sample standard deviation of round-10 accuracy across the three runs, and the paired clean-minus-attacked accuracy drop.

Three repeats assess variability; they do not establish implementation correctness. No new training runs were performed during the initial archive cleanup; subsequent controlled results are linked from INDEX.md across experiments/, studies/, Backdoor_test/, and Sybil_Backdoor_test/. The older runner still writes files in the repository root and can overwrite results on repeated invocations. The new runner requires a fresh output directory.

Run `python3 scripts/verify_result_catalog.py` to check the archived inventory.
