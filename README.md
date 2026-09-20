## Experiment records

See [the experiment data record](results/README.md) for the available baselines, archived results, known poisoning bug, and three-repeat protocol. [The catalog](results/catalog.csv) records original paths, validity labels, checksums, and final metrics. Existing raw data is preserved; no new experiment runs are included in this cleanup.

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




                2027 科展
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
              大量實驗證明有效
