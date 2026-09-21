# 程式閱讀指南

## 建議閱讀順序

1. `run_experiment_c.sh`：終端機指令如何交給 Python。
2. `run_experiment_c.py` 的 `main()`：先看 C 的整體流程，不必先讀完所有檢查細節。
3. `balanced_partition.py` 的 `balanced_partition()`：了解圖片如何分配，以及如何保證每端數量相同。
4. `run_balanced.py` 的 `BalancedClient` 和 `main()`：如何建立模型、配對訓練、保存結果。
5. `client.py` 的 `fit()`：真正的本地訓練與更新放大；C 沿用這些方法。
6. 回到 `run_experiment_c.py` 的 `validate_run()`、`summarize()`：理解檢查與統計。

## 各檔案負責什麼

| 檔案 | 責任 |
| --- | --- |
| `balanced_partition.py` | 純資料分配：輸入標籤、client 數量、Alpha、seed，回傳圖片索引 |
| `run_balanced.py` | 執行 C 的單一設定：三個 seeds，各做 Clean／Poisoned |
| `run_experiment_c.py` | 安排 C 的 pilot/full，驗證舊結果，再整理損害 |
| `run_matched.py` | A/B 的單一設定訓練，client 資料量不固定 |
| `run_study.py` | 安排 A/B 的設定組合並處理交會點去重 |
| `run_weight_comparison.py` | 實驗 D：沿用 A 的資料分配，比較樣本數加權與每端等權重 |
| `run_weight_comparison.sh` | 實驗 D 的安全 shell 入口；預設只預覽，`--run` 才訓練 |
| `source_compatibility.py` | 比較程式結構；允許純註解與排版不同，但仍拒絕運算或參數改動 |
| `scripts/report_matched.py` | A/B 單一設定的驗證與報告，不啟動訓練 |
| `scripts/build_science_report.py` | 核對 A/B 資料、畫圖與產生 Word，不啟動訓練 |
| `scripts/verify_result_catalog.py` | 確認歷史資料未被修改 |

## 常見變數

- `partitions[cid]`：某個 client 擁有的圖片索引；不是圖片本身。
- `capacity[cid]`：該 client 還可以接收多少張圖片。
- `preferences[label, cid]`：某類數字偏向分給該 client 的比例。
- `initial`：配對訓練開始前的共同模型權重。
- `parameters`：目前這一輪的全域模型權重。
- `updates`：各端送出的模型及樣本數，交給 FedAvg 合併。
- `paired_identity`：共同起點與資料分配的雜湊，避免比較到不同條件。
- `per_seed`／`rows`：各個 seed 的配對結果；之後才算平均與標準差。
- `accuracy_damage_pp`：Clean 準確率減 Poisoned 準確率，再乘 100；單位是百分點。
- `loss_damage`：Poisoned Loss 減 Clean Loss；正值代表變差。

## 這次可讀性整理的界線

加入繁體中文的函式說明、步驟註解，並拆開過長或擠在同一行的程式。訓練與分配程式的 Python 語法樹已與整理前比較，結構保持相同。

沿用檢查原本比較原始碼位元，連加註解都會被視為改版。現在改用 Python AST（語法樹）比較：只忽略註解、空白、引號排版等不改變結構的差異。常數、字串、文件字串、運算、函式呼叫順序與變數名稱仍會被比較。這不是可以自動認定任意兩個程式相同的工具。

保存結果的 SHA-256 檢查不變；**不要修改實驗資料夾內的 source 快照或 checksums.json**。本次不重新訓練、不改寫原始數據。
