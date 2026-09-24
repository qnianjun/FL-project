# 實驗 C 資料狀態

五組 Alpha 的配對訓練已完成，共 30 次條件訓練。完整結果看 [full/summary.csv](full/summary.csv) 與 [full/per_seed.csv](full/per_seed.csv)；跨階段入口看 [結果索引](../../INDEX.md)。

下表是原本的資料分配預檢紀錄，不是攻擊結果。

| Alpha | 平均標籤分布差異 TV | 各端樣本數 |
| --- | --- | --- |
| 0.01 | 0.507 | 每端 12,000 張；3 seeds 均通過 |
| 0.1 | 0.498 | 每端 12,000 張；3 seeds 均通過 |
| 1 | 0.251 | 每端 12,000 張；3 seeds 均通過 |
| 10 | 0.106 | 每端 12,000 張；3 seeds 均通過 |
| 100 | 0.038 | 每端 12,000 張；3 seeds 均通過 |

TV 越大代表各端的數字比例與全體比例差異較大。它不是攻擊損害；相鄰 Alpha 不保證在每個 seed 都按順序變化。

`pilot/` 是 Alpha 0.01、100 的部分統計，已包含於 `full/`，不能重複計數。各 `alpha_*_scale_10/` 保存原始結果，`preflight/` 保存分配檢查。操作與限制見 [EXPERIMENT_C.md](../../../docs/EXPERIMENT_C.md)。現有結果無須再次訓練。
