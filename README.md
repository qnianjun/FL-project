## Getting start
1. Clone the repository and set up you virtual environment.
```git clone "https://github.com/qnianjun/FL-project.git"```
```cd FL-project```
```python venv .venv```
```source .venv/bin/activate```
```pip install torch torchvision flwr```


2. Download the required dataset by runnuing.
```python3 -c "from torchvision import datasets, transforms; datasets.MNIST('./data', download=True)"```
