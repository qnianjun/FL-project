# 實驗 D：比較 FedAvg 的 client 權重

實驗 D 使用實驗 A、Alpha = 0.1 已保存的資料分配。它只改變模型合併方式：

- `weighted`：依每個 client 的樣本數加權，這是實驗 A 的方法。
- `equal`：五個 client 各占 20%，即使樣本數不同也一樣。

三個 seeds 下，攻擊者的資料量不同；因此這個實驗可以檢查 client 權重是否影響攻擊損害。Clean 和 Poisoned 仍然使用相同起始模型、同一份資料分配與相同洗牌 seed。

目前已有 weighted 結果，不需要重跑。先預覽：

```bash
cd /home/mark/FL-project
bash run_weight_comparison.sh
```

由你開始新的 equal-weight 訓練：

```bash
bash run_weight_comparison.sh --run
```

這會執行 6 次：3 seeds × Clean／Poisoned。輸出在：

```text
results/studies/weight_comparison/equal/
```

完成後，把 `summary.json`、`per_seed.csv` 和 `comparison.csv` 傳來。程式會直接讀取既有 weighted 結果，產生每個 seed 的比較表。比較公式是：

```text
accuracy damage = clean accuracy − poisoned accuracy
loss damage = poisoned loss − clean loss
```

若 equal-weight 的損害和 weighted 明顯不同，代表聚合權重可能是重要因素。若兩者接近，更新方向或資料分布可能更重要。這仍是三個 seeds 的初步證據。

程式會拒絕已存在的輸出資料夾，避免覆寫。它不接續 checkpoint；六次訓練都從各自配對的初始模型開始。
