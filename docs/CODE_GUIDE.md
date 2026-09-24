# 檔案與程式指南


## 資料夾用途

| 位置 | 放什麼／何時閱讀 |
| --- | --- |
| 專案根目錄的 `client.py`、`run_*.py` | 共用模型與各實驗執行器；保留原位置，避免破壞匯入及來源核對 |
| `docs/` | 設計、操作、結果解讀；最新結論看 [ATTACK_RESULTS_SUMMARY.md](ATTACK_RESULTS_SUMMARY.md) |
| `results/` | 原始訓練資料與彙總；依 [INDEX.md](../results/INDEX.md) 找正式採用結果 |
| `reports/current/` | 舊階段的 Model poisoning A/B 報告；名稱不代表所有階段都已包含 |
| `reports/backdoor/` | Backdoor 的 Word、PDF、圖表與核對紀錄 |
| `scripts/` | 報告生成與歷史資料驗證，不是完整訓練入口 |
| `tests/` | 快速回歸測試，使用小型資料驗證訓練／配對／沿用規則 |
| `data/` | MNIST 快取；不是實驗輸出 |
| `.venv/` | 本機 Python 環境；重現版本請看目標實驗的 `environment.json` |
| `.agents/skills/` | 專案協作規範，例如 Python 可讀性要求 |
| `config.json` | 目前執行用設定，不等於所有歷史實驗設定 |
| `TISF.docx` | 原科展格式參考；不是本專案訓練結果 |

## 依研究階段找程式

| 階段 | 入口／核心 | 說明 |
| --- | --- | --- |
| A/B | `run_study.py` → `run_matched.py` → `client.py` | 原始不同資料量的 Model-update poisoning |
| C | `run_experiment_c.py` → `run_balanced.py`、`balanced_partition.py` | 固定每端 12,000 張的 Model-update poisoning |
| D | `run_weight_comparison.py` | 沿用 A 的原始分配，只改模型合併規則 |
| Backdoor pilot | `run_backdoor.py` | 重用 BalancedClient 與訓練函式，新增觸發圖片及 ASR |
| Backdoor Alpha 比較 | `run_backdoor_sweep.py` | 驗證 pilot 沿用，安排其他 Alpha，整理配對結果 |
| Sybil-Backdoor | `run_sybil_backdoor.py` | 重複 client 0 的身分及資料；控制同一批毒化圖片與排列順序 |
| A/B 報告 | `scripts/build_science_report.py` | 讀取保存結果，產生文件與圖表 |
| Backdoor 報告 | `scripts/build_backdoor_report.py` | 讀取保存結果，產生高中生用語版文件與圖表 |

`.sh` 檔是對應 Python 執行器的終端機包裝。舊 `run_all.py`、`run_all.sh`、`run.sh` 與 `server.py` 保留作早期流程參考，不能當成目前受控配對實驗的替代入口。`results/accuracy.py`、`results/loss.py` 是舊繪圖工具，正式比較請使用各階段已驗證的彙總資料。

目前各階段已完成，讀取結果不需要再次執行訓練。若要增加實驗，先看階段文件及輸出目錄規則；Backdoor／Sybil 的現有預設輸出已存在，直接使用原預設預覽也可能被拒絕。

## 建議閱讀順序（先理解共用訓練與 C）

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

## 既有可讀性與來源相容性規則

先前的可讀性整理加入中文說明並調整排版；當時以 Python 語法樹核對訓練與分配邏輯。本次檔案整理只更新導覽文件，不變更訓練程式。

沿用檢查原本比較原始碼位元，連加註解都會被視為改版。部分舊實驗沿用檢查改用 Python AST（語法樹）比較：只忽略註解、空白、引號排版等不改變結構的差異。常數、字串、文件字串、運算、函式呼叫順序與變數名稱仍會被比較。這不是可以自動認定任意兩個程式相同的工具。

Backdoor／Sybil 沿用另外要求原始碼與環境精確相符，不能概括套用上述較寬鬆規則。保存結果的 SHA-256 檢查不變；**不要修改實驗資料夾內的 source 快照或 checksums.json**。本次不重新訓練、不改寫原始數據。
