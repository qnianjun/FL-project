import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =========================
# 設定
# =========================

NUM_CLIENTS = 5

# alpha 越大 → 越接近 IID
# alpha 越小 → Non-IID 越嚴重
ALPHA = 1.0

# 固定亂數，讓實驗可以重現
SEED = 42


# =========================
# Model
# =========================

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# =========================
# 建立 Dirichlet Partition
# =========================

def create_dirichlet_partition(dataset, num_clients, alpha):
    """
    使用 Dirichlet distribution
    將 MNIST 分配給不同 Client。

    每一筆資料只會屬於一個 Client。
    """

    np.random.seed(SEED)

    targets = np.array(dataset.targets)

    client_indices = [[] for _ in range(num_clients)]

    # MNIST 有 10 個類別
    num_classes = 10

    for class_id in range(num_classes):

        # 找出這個 class 的所有資料
        class_indices = np.where(targets == class_id)[0]

        # 打亂
        np.random.shuffle(class_indices)

        # Dirichlet 分配比例
        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # 根據比例計算每個 Client 要拿多少資料
        proportions = (
            np.cumsum(proportions) * len(class_indices)
        ).astype(int)

        proportions = np.diff(
            np.concatenate(([0], proportions))
        )

        start = 0

        for client_id, count in enumerate(proportions):

            end = start + count

            client_indices[client_id].extend(
                class_indices[start:end]
            )

            start = end

    # 最後再打亂每個 Client 的資料
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


# =========================
# Load Data
# =========================

def load_data():

    transform = transforms.ToTensor()

    trainset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transform
    )

    return trainset, testset


# =========================
# Train
# =========================

def train(model, trainloader):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01
    )

    model.train()

    for images, labels in trainloader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()


# =========================
# Test
# =========================

def test(model, testloader):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in testloader:

            outputs = model(images)

            batch_loss = criterion(
                outputs,
                labels
            )

            loss += batch_loss.item() * len(labels)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += len(labels)

            correct += (
                predicted == labels
            ).sum().item()

    loss = loss / total

    accuracy = correct / total

    return loss, accuracy


# =========================
# Flower Client
# =========================

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid):

        self.cid = int(cid)

        self.model = Net()

        trainset, testset = load_data()

        # 建立 Dirichlet partition
        partitions = create_dirichlet_partition(
            trainset,
            NUM_CLIENTS,
            ALPHA
        )

        # 取得這個 Client 的資料
        client_indices = partitions[self.cid]

        self.trainset = Subset(
            trainset,
            client_indices
        )

        self.trainloader = DataLoader(
            self.trainset,
            batch_size=32,
            shuffle=True
        )

        self.testset = testset

        self.testloader = DataLoader(
            self.testset,
            batch_size=32,
            shuffle=False
        )

        print(
            f"Client {self.cid}: "
            f"{len(self.trainset)} training samples"
        )


    # =====================
    # Get Parameters
    # =====================

    def get_parameters(self, config):

        return [
            value.cpu().numpy()
            for value in self.model.state_dict().values()
        ]


    # =====================
    # Set Parameters
    # =====================

    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )

        state_dict = {
            key: torch.tensor(value)
            for key, value in params_dict
        }

        self.model.load_state_dict(
            state_dict,
            strict=True
        )


    # =====================
    # Fit
    # =====================

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        train(
            self.model,
            self.trainloader
        )

        return (
            self.get_parameters(config={}),
            len(self.trainset),
            {}
        )


    # =====================
    # Evaluate
    # =====================

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        loss, accuracy = test(
            self.model,
            self.testloader
        )

        return (
            float(loss),
            len(self.testset),
            {
                "accuracy": float(accuracy)
            }
        )


# =========================
# Start Client
# =========================

if __name__ == "__main__":

    cid = sys.argv[1]

    client = FlowerClient(cid)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )
