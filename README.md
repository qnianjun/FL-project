## Getting start
1. Clone the repository and set up you virtual environment. <br>
```git clone "https://github.com/qnianjun/FL-project.git"``` <br>
```cd FL-project``` <br>
```python -m venv .venv``` <br>
```source .venv/bin/activate``` <br>
```pip install torch torchvision flwr``` <br>


2. Download the required dataset by runnuing. <br>
```python3 -c "from torchvision import datasets, transforms; datasets.MNIST('./data', download=True)"```
