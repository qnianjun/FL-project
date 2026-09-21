# 資料總覽與下一階段

## 可以用於目前 A/B 報告的資料

| 資料 | 位置 | 狀態 |
| --- | --- | --- |
| A/B 全部設定統計 | [summary.csv](studies/alpha_scale/summary.csv) | 已完成：9 個不同設定，54 次訓練 |
| 每個 seed 的配對結果 | [per_seed.csv](studies/alpha_scale/per_seed.csv) | 保留個別差異，不只看平均 |
| 原始結果所在位置 | [study.json](studies/alpha_scale/study.json) | A/B 交會設定指向同一批資料 |
| Word 與圖表 | [reports/current](../reports/current/README.md) | 根據已完成 A/B 資料撰寫的階段報告 |

部分已完成資料在 `experiments/2026-09-20_alpha_*/`，其餘在 `studies/alpha_scale/alpha_*/`；以 study.json 為準。為維持原始來源紀錄，這次不移動原始路徑。

## 實驗 C：固定每端資料量

新結果放在 `studies/equal_samples/`，與 A/B 分開。`preflight/` 只是資料分配檢查，**不代表已完成訓練**。`pilot/summary.csv` 和 `full/summary.csv` 只在對應階段的訓練及驗證完成後產生。

操作說明：[EXPERIMENT_C.md](../docs/EXPERIMENT_C.md)。

## 保留但不混入主分析的資料

實驗 D 的程式已加入，但尚未產生新的等權重訓練結果。它會沿用 A 的 Alpha = 0.1 資料分配，新增 3 seeds × Clean／Poisoned，共 6 次訓練。說明見 [EXPERIMENT_D.md](../docs/EXPERIMENT_D.md)。

- `baselines/initial_alpha_sweep/`：早期單次 Clean 結果，只作初步參考。
- `archive/`：歷史原始資料；舊版 poisoning 受程式錯誤影響，不作攻擊效力證據。
- `experiments/manual_unverified/`：來源設定未完整確認的手動執行結果。
- `catalog.csv`：歷史檔案目錄；其 58 份 CSV 包含重複內容，不是 58 個獨立實驗。

不刪除舊資料，不把相同 seeds 的重跑當新的獨立 seeds。此整理未啟動完整訓練。
