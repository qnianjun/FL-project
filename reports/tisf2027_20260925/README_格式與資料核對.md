# 2027 臺灣國際科學展覽會研究報告：交付與核對

本資料夾內以「電腦科學與資訊工程科_」開頭的 Word、PDF 為研究報告。新版共18頁（含封面）、10張圖、10張表。其他檔案供研究者查核，不是報名必須一併上傳的附件。

## 格式依據與完成項目

- 已閱讀使用者提供的2027報名注意事項PDF，並核對[官方下載頁](https://twsf.ntsec.gov.tw/Article.aspx?a=36&lang=1)及[臺灣國際科學展覽會實施要點](https://edu.law.moe.gov.tw/LawContent.aspx?id=FL030579)。
- 封面使用2027官方附件五樣式；保留區別及編號空白，依主辦單位通知處理。作品名稱、科別與三個關鍵詞已填入。
- 依附件四採「壹、前言」「貳、研究方法或過程」「參、研究結果與討論」「肆、結論與應用」「伍、參考文獻」，附中英文摘要，参考文獻採APA格式。
- A4；Word實際統計摘要為中文277字、英文171字。中文逐字元含標點計算319，兩種計法均未超過350。匿名本文、封面及檔案中繼資料已檢查，無研究資料網址，圖表均交代來源。
- Word約0.89 MB、PDF約1.14 MB，均小於10 MB；Microsoft Word實際轉PDF後檢查分頁，沒有空白頁。
- 本次選用正文12點、1.5倍行距及約2.3–2.5公分邊界以利閱讀；未將這些排版選擇宣稱為主辦單位明訂數值。所查文件未明訂本報告的頁數上限。
- 去年TISF.docx僅用於理解章節與呈現風格；其研究題材與本計畫不同，未引用其研究數據或成果。

報告與報名其他材料不同：仍須依學校／主辦單位程序提供實際研究日誌等應備文件；本次未代造研究日誌，也未提交報名。

## 圖表清單

1. 研究流程示意（方法圖，非新增實驗）。
2. alpha=0.1、seed=42的標籤分布熱圖：清楚呈現三個Sybil共用一份資料。
3. MNIST原圖與右下角3×3白色觸發圖樣示例。
4. 一般分割與等量分割下的模型更新投毒損害。
5. 後門alpha掃描與Sybil身分數比較。
6. 三類攻擊的FedAvg、裁剪與中位數防禦比較。
7. 正常準確率與後門ASR的取捨。
8. 相同／不同洗牌的配對更新相似度分布。
9. 三身分後門逐輪正常準確率及ASR。
10. 不同洗牌各種子的逐輪Sybil相似度及固定門檻。

誤差棒／陰影的定義逐圖註明；三種子樣本標準差不是信賴區間，也不代表正式顯著性檢定。圖10的陰影是每輪三組配對的最小至最大值。

## 資料依據

- 模型更新投毒A、B：`results/studies/alpha_scale/`。
- 等量分割C：`results/studies/equal_samples/full/`。
- 聚合權重控制D：`results/studies/weight_comparison/equal/`。
- seed 43稽核：`results/experiments/repro_alpha_0.1_scale_10_seeds_42_43/`。
- 後門alpha掃描：`results/Backdoor_test/alpha_sweep/`。
- Sybil身分數：`results/Sybil_Backdoor_test/alpha_0.1_ratio_0.2/`。
- 三種基準防禦：`results/defense_baselines/pilot_alpha_0.1/`。
- 原始更新相似度診斷：`results/defense_diagnostics/sybil_update_similarity_rerun/`。
- 相同洗牌分組：`results/defense_baselines/clip_similarity_group_pilot/`。
- 不同洗牌穩健性：`results/defense_baselines/sybil_distinct_shuffle/`。
- 圖2讀取保存的partition索引及MNIST原始標籤；圖3僅複製MNIST測試影像供圖解，未修改資料集。圖9直接讀取已完成的每輪results.csv，圖10讀取已保存的pairwise_similarity_labeled.csv。

核對包括：結果清單中1,515個唯一檔案雜湊、防禦最後一輪與per_seed的一致性、summary的平均與樣本標準差、配對條件的初始化／分割／洗牌／樣本數，以及分組逐輪紀錄。詳見`input_validation.json`、`format_validation.json`。沒有執行新訓練，沒有修改raw results。

## 解讀界線

不同洗牌下ASR回升至98.94%，為本報告的重要負面結果。它直接檢验同一分組方法的洗牌穩健性；與其他相同洗牌聚合基準相比則同時改變洗牌與方法，不能宣稱是純方法效果。近重複門檻未重新調整，不宣稱通用Sybil辨識或防禦。

報告產生程式：`scripts/build_tisf2027_report.py`。程式僅分析保存資料、繪圖與產生文件；最終PDF由Microsoft Word輸出並檢查。原始Word、規範PDF與所有實驗結果保持原狀。
