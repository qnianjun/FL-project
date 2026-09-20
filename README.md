## Experiment records

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
