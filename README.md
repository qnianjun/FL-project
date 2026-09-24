# 聯邦學習科展專案

研究不同資料分配、模型更新放大、後門圖樣及重複攻擊者身分，如何影響 MNIST 聯邦學習。目前已完成 Model-update poisoning A/B/C/D、seed-43 重現檢查、Backdoor 與 Sybil-Backdoor；防禦方法尚未納入這批結果。

## 從這裡開始

| 想做什麼 | 入口 |
| --- | --- |
| 看各種攻擊的主要發現與研究限制 | [攻擊結果摘要](docs/ATTACK_RESULTS_SUMMARY.md) |
| 找正式結果、每個 seed 的數值及來源 | [訓練結果索引](results/INDEX.md) |
| 了解資料夾與程式用途 | [檔案與程式指南](docs/CODE_GUIDE.md) |
| 閱讀 Model poisoning A/B 階段報告 | [Word](reports/current/聯邦學習_目前研究報告.docx)／[PDF](reports/current/聯邦學習_目前研究報告.pdf) |
| 閱讀 Backdoor 階段報告（高中生用語版） | [Word](reports/backdoor/聯邦學習_Backdoor階段報告.docx)／[PDF](reports/backdoor/聯邦學習_Backdoor階段報告.pdf) |
| 查看舊版資料與排除原因 | [歷史資料紀錄](results/README.md) |

`reports/current/` 的「current」是原 A/B 報告的既有目錄名稱，不代表它包含後來的 C/D、Backdoor 或 Sybil。跨階段最新整理以攻擊結果摘要為準。

## 已完成的研究

| 階段 | 比較內容 | 結果位置 |
| --- | --- | --- |
| Model-update poisoning A/B | 改變 Alpha／更新放大倍率 Scale | [A/B 統計](results/studies/alpha_scale/summary.csv) |
| C | 固定每端 12,000 張，再比較 Alpha | [C 完整統計](results/studies/equal_samples/full/summary.csv) |
| D | 同一份資料分配，改變合併權重 | [逐 seed 權重比較](results/studies/weight_comparison/equal/comparison.csv) |
| seed-43 重現檢查 | 重做 seeds 42、43 的原始 weighted 設定 | [重現結果](results/experiments/repro_alpha_0.1_scale_10_seeds_42_43/summary.json) |
| Backdoor | 固定圖樣、目標 0、修改攻擊端 20% 圖片；比較 Alpha | [Backdoor 統計](results/Backdoor_test/alpha_sweep/summary.csv) |
| Sybil-Backdoor | 不增加攻擊者獨有資料，改成 1／2／3 個身分 | [Sybil 統計](results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/summary.csv) |

Model-update poisoning 是放大模型更新；Backdoor 才修改圖片與訓練標籤。Equal weighting 是比較條件，不是已驗證的防禦。結果解讀與限制詳見摘要。

## 檢查與使用

在專案根目錄使用現有 `.venv`：

```bash
# 快速測試；使用小型測試資料，不跑完整 MNIST 聯邦訓練
.venv/bin/python -m unittest discover -s tests -v

# 核對歷史 catalog；它不是所有新階段的總目錄
python3 scripts/verify_result_catalog.py
```

實驗操作說明：[原 A/B 設計](docs/STUDY_PLAN.md)、[C](docs/EXPERIMENT_C.md)、[D](docs/EXPERIMENT_D.md)、[Backdoor](docs/BACKDOOR_SWEEP.md)、[Sybil-Backdoor](docs/SYBIL_BACKDOOR.md)。現有預設結果目錄已完成，無須為了看結果再次訓練；部分執行器連預覽也會拒絕已存在的輸出目錄。

重建環境時，套件版本以目標實驗的 `environment.json` 為準。`data/` 是 MNIST 快取，`.venv/` 是本機 Python 環境。根目錄 `config.json` 是執行設定，不能用來推定過去每次實驗的參數。

## 檔案保存原則

程式維持既有匯入位置，原始結果維持既有路徑。不要更動實驗資料夾中的 CSV、config、partition、source 快照或 checksums；若要重跑，使用新輸出目錄。舊入口 `run_all.py`／`run_all.sh`／`run.sh` 留作歷史用途，不是目前受控實驗的建議入口。
