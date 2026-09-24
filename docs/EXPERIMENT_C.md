# 實驗 C（已完成：固定每端資料量）

目前五組 Alpha、30 次配對條件訓練已完成。請直接讀取 [full/summary.csv](../results/studies/equal_samples/full/summary.csv) 與 [full/per_seed.csv](../results/studies/equal_samples/full/per_seed.csv)，無須重跑。

這裡的「繼續訓練」是繼續研究的新實驗，每次從控制好的初始模型開始，**不是接續舊模型的 checkpoint**。

## 研究設計

- 五個 client 各 12,000 張，合計 60,000 張且不重複、不遺漏。
- Alpha：0.01、0.1、1、10、100；Scale 固定 10。
- 攻擊者 client 0；seeds 42、43、44；Clean／Poisoned 配對。
- 每次十輪、每輪本地訓練一遍、batch size 32、SGD learning rate 0.01。
- 每端 FedAvg 權重固定為 20%；模型與攻擊方法沿用已修正的 client.py。

## 原始執行流程（已完成，供重現時參考）

```bash
cd /home/mark/FL-project
# 只看計畫，不會訓練或寫入結果
bash run_experiment_c.sh

# 先檢查 Alpha 0.01 與 100 的資料分配，產生數字分布圖；不訓練
bash run_experiment_c.sh --check

# 小範圍正式訓練：2 個 Alpha × 3 seeds × Clean/Poisoned = 12 次
bash run_experiment_c.sh --run

# 完成後補齊全部 Alpha；已完成的小範圍設定會驗證後沿用
bash run_experiment_c.sh --phase full --run
```

全套共 30 次訓練；若先完成 12 次，補齊時剩 18 次。執行中保持終端機開啟。

檢查所有 Alpha 的資料分配（仍不訓練）：

```bash
bash run_experiment_c.sh --phase full --check
```

## 看哪些檔案？

```text
results/studies/equal_samples/
├── preflight/pilot/ 或 full/  # 數字分布圖、樣本數、分配雜湊、TV 數值
├── alpha_0.01_scale_10/       # 每個設定的原始訓練結果
├── alpha_100_scale_10/
├── pilot/summary.csv         # 小範圍完成後的統計
└── full/summary.csv          # 全部完成後的統計
```

`full/summary.csv` 已包含全部五組；`pilot/summary.csv` 是其中兩組的子集，不能重複計數。各階段也有 `per_seed.csv`、`damage.png` 與 `REPORT.md`。不需用 A/B 的 report_matched.py 產生 C 的報告。

已完成的設定會檢查原始檔案雜湊、訓練程式、配對識別資料、每端數量與更新倍率。若資料夾存在但訓練未完成，會停止，**不覆寫**。請保留錯誤訊息；必要時用 `--output results/studies/equal_samples_retry` 建立另一批，不能把半成品當完成品。

## 分配方法的差異必須說清楚

原本 A 使用無固定容量的 Dirichlet 分配。C 使用 `capacity_constrained_dirichlet_v1`：

1. 每個數字類別抽出五個 client 的偏好比例。
2. 隨機排列所有訓練圖片，依類別偏好分配。
3. 已滿 12,000 張的 client 不再收資料；必要時在仍有空位的 client 中分配。
4. 每張圖只使用一次，不截掉資料、不重複抽樣補滿。

容量限制會改變實際分布，因此 **相同 Alpha 不代表 A、C 有相同的標籤差異程度**。同一 seed 在兩種演算法中也不代表相同資料分配。C 的內部 Clean／Poisoned 才是完全相同資料的配對。

預檢圖顯示各端數字比例，另保存 TV（各端標籤比例與全體標籤比例的平均差異，範圍 0～1）：越高表示標籤分布越不同。這是描述資料分布的數值，不是攻擊損害。用實際分布檢查 Alpha 的效果，不把「小 Alpha 一定更不均」當成每個 seed 的保證。

比較 A 與 C 可以提供線索，但演算法也改了，不能把所有結果差異都歸因於 client 權重。已完成的 [實驗 D](EXPERIMENT_D.md) 沿用同一份分配來比較權重，是較直接的合併規則對照。
