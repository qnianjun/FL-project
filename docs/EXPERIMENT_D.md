# 實驗 D：比較 FedAvg 的 client 權重

實驗 D 使用實驗 A、Alpha = 0.1 已保存的資料分配。它只改變模型合併方式：

- `weighted`：依每個 client 的樣本數加權，這是實驗 A 的方法。
- `equal`：五個 client 各占 20%，即使樣本數不同也一樣。

三個 seeds 下，攻擊者的資料量不同；因此這個實驗可以檢查 client 權重是否影響攻擊損害。Clean 和 Poisoned 仍然使用相同起始模型、同一份資料分配與相同洗牌 seed。

weighted 與 equal 結果皆已完成。weighted 沿用 A 的六次訓練，equal 新增六次；目前不需要重跑。

- [comparison.csv](../results/studies/weight_comparison/equal/comparison.csv)：逐 seed 的 weighted／equal 比較。
- [summary.json](../results/studies/weight_comparison/equal/summary.json)：equal 統計。
- [per_seed.csv](../results/studies/weight_comparison/equal/per_seed.csv)：equal 每個 seed 的結果。
- [environment.json](../results/studies/weight_comparison/equal/environment.json)：`source_partition` 指向沿用的 A 資料。

| Seed | Weighted 準確率損害（百分點） | Equal 準確率損害（百分點） |
| --- | --- | --- |
| 42 | 56.32 | 31.02 |
| 43 | −0.04 | 26.31 |
| 44 | 46.38 | 44.43 |

Equal weighting 是研究對照，不是已證明有效的防禦。平均損害相近，但個別 seed 的變化方向不同，不能只看平均。完整解讀見 [攻擊結果摘要](ATTACK_RESULTS_SUMMARY.md)。

比較公式是：

```text
accuracy damage = clean accuracy − poisoned accuracy
loss damage = poisoned loss − clean loss
```

這個對照重用相同分配、起始模型與洗牌設定，比 A 對 C 更直接控制資料差異。但它同時改變所有 client 的權重，且只有三個 seeds，不能宣稱統計顯著或只歸因於攻擊者權重。

程式會拒絕已存在的輸出資料夾，避免覆寫。它不接續 checkpoint；六次訓練都從各自配對的初始模型開始。
