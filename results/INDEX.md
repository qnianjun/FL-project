# 訓練結果索引

這是目前各階段的結果入口；結論請看 [攻擊結果摘要](../docs/ATTACK_RESULTS_SUMMARY.md)。原始資料不搬移、不合併、不覆寫，以保留來源路徑與雜湊紀錄。

## 1. 正式採用的完成結果

「一次訓練」指一個 seed、一個條件的十輪訓練；不是十個獨立實驗。除重現檢查外，每個設定均使用 seeds 42、43、44，各做無攻擊／有攻擊配對。

| 階段 | 設定與完成範圍 | 彙總 | 每個 seed／來源 | 本索引去除沿用後採計的訓練次數 |
| --- | --- | --- | --- | --- |
| Model-update poisoning A/B | A：5 個 Alpha，Scale 10；B：5 個 Scale，Alpha 0.1；交會點只計一次 | [summary.csv](studies/alpha_scale/summary.csv) | [per_seed.csv](studies/alpha_scale/per_seed.csv)／[study.json](studies/alpha_scale/study.json) | 54 |
| C：各端等量 | 5 個 Alpha；每端 12,000 張；Scale 10 | [full/summary.csv](studies/equal_samples/full/summary.csv) | [full/per_seed.csv](studies/equal_samples/full/per_seed.csv) | 30 |
| D：合併權重對照 | Alpha 0.1、Scale 10；沿用 A 的 weighted 六次，新增 equal 六次 | [summary.json](studies/weight_comparison/equal/summary.json)／[comparison.csv](studies/weight_comparison/equal/comparison.csv) | [per_seed.csv](studies/weight_comparison/equal/per_seed.csv)／[來源紀錄](studies/weight_comparison/equal/environment.json) | 6 |
| 重現檢查 | 原 weighted、Alpha 0.1、Scale 10；重做 seeds 42、43 | [summary.json](experiments/repro_alpha_0.1_scale_10_seeds_42_43/summary.json) | [來源紀錄](experiments/repro_alpha_0.1_scale_10_seeds_42_43/environment.json) | 4（不是新 seeds） |
| Backdoor pilot | Alpha 0.1；固定白色圖樣、目標 0、修改比例 20% | [summary.json](Backdoor_test/mnist_alpha_0.1_ratio_0.2/summary.json) | [config.json](Backdoor_test/mnist_alpha_0.1_ratio_0.2/config.json) | 6 |
| Backdoor Alpha 比較 | 5 個 Alpha；沿用 pilot 六次，新增四組 | [summary.csv](Backdoor_test/alpha_sweep/summary.csv) | [per_seed.csv](Backdoor_test/alpha_sweep/per_seed.csv)／[provenance.json](Backdoor_test/alpha_sweep/provenance.json) | 24 |
| Sybil-Backdoor | 1／2／3 個攻擊者身分；沿用單身分 pilot，新增兩組 | [summary.csv](Sybil_Backdoor_test/alpha_0.1_ratio_0.2/summary.csv) | [per_seed.csv](Sybil_Backdoor_test/alpha_0.1_ratio_0.2/per_seed.csv)／[provenance.json](Sybil_Backdoor_test/alpha_0.1_ratio_0.2/provenance.json) | 12 |

以上主分析採用 **132 次條件訓練結果，另有 4 次重現檢查**，共 136 次；不是 136 個獨立 seeds，也不是磁碟上所有歷史執行的總數。Backdoor 掃描總共含 30 次，Sybil 表格總共含 18 次，但兩者的 pilot 沿用不能再算成新訓練。

## 2. 原始結果怎麼找

- **A/B：**以 `studies/alpha_scale/study.json` 的 `entries[].path` 為準。檔案分布於 `experiments/2026-09-20_alpha_*/` 與 `studies/alpha_scale/alpha_*/`。未被 manifest 選用的同名參數結果也保留，但不要混進同一批統計或當額外獨立 seeds。
- **C：**`studies/equal_samples/alpha_*_scale_10/` 是五組原始資料；`pilot/` 與 `full/` 是部分／全部結果的整理，不能再相加。`preflight/` 只有資料分配檢查，不是模型訓練。
- **D：**`studies/weight_comparison/equal/` 保存新增的等權重結果。分配索引依 `environment.json` 的 `source_partition` 指向 A 的檔案；沒有獨立的 `partitions.npz` 副本是原設計。
- **Backdoor：**`mnist_alpha_0.1_ratio_0.2/` 是原 pilot；`alpha_sweep/alpha_0.1/` 是沿用副本，其餘 `alpha_*/` 為其他設定。
- **Sybil：**`Sybil_Backdoor_test/alpha_0.1_ratio_0.2/sybil_1/` 也是 pilot 副本；`sybil_2/`、`sybil_3/` 是新增結果。`Sybil_test/` 目前是空目錄，別與已完成結果混淆。

每個條件通常放在 `seed_42/clean/`、`seed_42/poisoned/` 或 `seed_42/backdoor/`。先讀 `config.json`，再讀 `results.csv`；需要查來源時看設定根目錄的 `environment.json`、`source/`、`dataset_hashes.json` 與 `checksums.json`。Sybil 新設定的共用來源資料在整個 Sybil 實驗根目錄。

## 3. 哪些檔案拿來做什麼

| 檔案 | 用途 |
| --- | --- |
| `summary.csv`／`summary.json` | 各設定的平均、樣本標準差或完整條件摘要 |
| `per_seed.csv` | 各 seed 最後一輪與配對差，避免只看平均 |
| `comparison.csv` | D 的 weighted／equal 逐 seed 對照 |
| `results.csv` | 每輪的原始指標；不是每列一個獨立實驗 |
| `updates.csv` | Model poisoning 的更新大小與放大檢查 |
| `partitions.npz` | 真正分配的圖片索引；Sybil 需搭配身分對應表閱讀 |
| `config.json` | 該次實驗的參數、配對雜湊與樣本數 |
| `source/`、`environment.json`、`provenance.json` | 當時程式、版本、沿用關係；不能用目前程式取代快照 |
| `checksums.json` | 核對保存內容；不能修改結果後自行重算來假裝原始結果未變 |

指標方向不同：Model poisoning 的準確率損害是 **Clean − Poisoned**；Backdoor／Sybil 的準確率與 ASR 差是 **Backdoor − Clean**。準確率有的原始檔使用 0～1，有的使用百分比，讀欄名與設定後再整理。

## 4. 報告與歷史資料

- [跨攻擊摘要](../docs/ATTACK_RESULTS_SUMMARY.md)：涵蓋 A/B/C/D、重現檢查、Backdoor、Sybil，含限制與防禦測試建議。
- [A/B 報告](../reports/current/README.md)：原 Model poisoning 階段的 Word、PDF 與圖；不是全部階段的總報告。
- [Backdoor 報告](../reports/backdoor/README.md)：高中生用語版 Word／PDF，含 pilot 與 Alpha 比較。
- `baselines/initial_alpha_sweep/`：早期單次正常訓練，只作初步參考。
- `archive/`、[catalog.csv](catalog.csv)：歷史來源與檔案核對。舊版 poisoning 的參數快照錯誤已記錄，排除於目前攻擊效果分析。
- `experiments/manual_unverified/`：設定或來源尚未完整確認，不併入正式三個 seeds 的統計。
- `experiments/manual_alpha_0.1/` 及舊 `Dirichlet_*` 路徑目前只剩空目錄結構，不能視為完成結果。

本次整理只更新導覽與進度文件，未重新訓練，也未更動原始實驗檔案。完整的舊資料說明仍保留在 [results/README.md](README.md)。
