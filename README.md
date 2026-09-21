## Experiment records

閱讀程式可先看 [程式閱讀指南](docs/CODE_GUIDE.md)，依序了解 shell 入口、資料分配、訓練、驗證與報告。

目前 A/B 已完成九個不同設定、54 次訓練，請先看 [資料總覽](results/INDEX.md)。下一階段是 [實驗 C：固定每端 12,000 張](docs/EXPERIMENT_C.md)。預覽：`bash run_experiment_c.sh`；資料分配檢查：`bash run_experiment_c.sh --check`；由你開始小範圍訓練：`bash run_experiment_c.sh --run`。A/B 原始結果與程式保留，新結果寫入獨立資料夾。

實驗 D 比較樣本數加權與每端等權重的 FedAvg，使用 A 的 Alpha = 0.1 資料分配。已有的加權結果會沿用，只需執行新的等權重六次配對訓練：`bash run_weight_comparison.sh --run`。詳見 [實驗 D 說明](docs/EXPERIMENT_D.md)。

The current [research design](docs/STUDY_PLAN.md) has two experiments: **A varies alpha** (0.01, 0.1, 1, 10, 100) with scale 10; **B varies scale** (1, 2, 5, 10, 20) with alpha 0.1. Each setting uses seeds 42, 43, and 44, matched clean/poisoned runs, and accuracy/loss damage measurements.

Preview with `bash run_study.sh --reuse-existing`. Start training yourself with `bash run_study.sh --reuse-existing --run`. The two existing validated settings can be reused, leaving 42 new training runs. The default output is `results/studies/alpha_scale/`.

See [the experiment data record](results/README.md) for the available baselines, archived results, known poisoning bug, and three-repeat protocol. [The catalog](results/catalog.csv) records original paths, validity labels, checksums, and final metrics. Existing raw data is preserved. New matched experiments are stored separately under `results/experiments/`.

Usable controlled results now cover **alpha 0.01 and 0.1 at poisoning scale 10**, each with three matched clean/poisoned repetitions (12 runs total). See [the results guide](results/README.md#data-usable-now) for the comparison and limits. Historical baselines are preliminary; old buggy poisoning results are excluded from attack analysis.

Run three paired clean/poisoned repeats with `.venv/bin/python run_matched.py --output results/experiments/NEW_RUN`. This uses seeds 42, 43, and 44, alpha 0.1, and scale 10 by default; the output directory must not already exist. Generate the validated report with `.venv/bin/python scripts/report_matched.py results/experiments/NEW_RUN`.

Verify the archive with `python3 scripts/verify_result_catalog.py` and the parameter snapshot fix with `.venv/bin/python -m unittest discover -s tests -v`.

## Getting start
1. Clone the repository and set up you virtual environment. <br>
```git clone "https://github.com/qnianjun/FL-project.git"``` <br>
```cd FL-project``` <br>
```python -m venv .venv``` <br>
```source .venv/bin/activate``` <br>
```pip install torch torchvision flwr``` <br>


2. Download the required dataset by runnuing. <br>
```python3 -c "from torchvision import datasets, transforms; datasets.MNIST('./data', download=True)"```


```             2027 科展
                    │
                    ▼
             Federated Learning
                    │
                    ▼
                Non-IID
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Poisoning    Backdoor     Sybil
        │           │           │
        └───────────┼───────────┘
                    ▼
                做實驗比較
                    │
                    ▼
               找出最有問題的
                    │
                    ▼
               Research Gap
                    │
                    ▼
                自己的方法
                    │
                    ▼
              大量實驗證明有效```
