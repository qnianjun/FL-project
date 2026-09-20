import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

#------------多次-----------------#

# =========================
# 實驗設定
# =========================
with open("config.json", "r") as f:
    cfg = json.load(f)

NUM_CLIENTS = 5

ENABLE_POISON = cfg["enable_poison"]

POISON_CLIENT = cfg["poison_client"]

POISON_SCALE = cfg["poison_scale"]

ALPHA = cfg["alpha"]

SEED = 42

#------------單次-----------------#
# # =========================
# # 實驗設定
# # =========================

# NUM_CLIENTS = 5


# # ---------- Poisoning ----------

# ENABLE_POISON = False

# POISON_CLIENT = 0
# POISON_SCALE = 10

# # ---------- Dirichlet ----------

# ALPHA = 0.1

# # ---------- Random Seed ----------

# SEED = 42


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

    client_indices = [
        [] for _ in range(num_clients)
    ]

    num_classes = 10

    for class_id in range(num_classes):

        # 找出這個 class 的所有資料

        class_indices = np.where(
            targets == class_id
        )[0]

        # 打亂

        np.random.shuffle(class_indices)

        # Dirichlet 分配比例

        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # 根據比例計算資料數量

        proportions = (
            np.cumsum(proportions)
            * len(class_indices)
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

        np.random.shuffle(
            client_indices[client_id]
        )

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

        loss = criterion(
            outputs,
            labels
        )

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

            loss += (
                batch_loss.item()
                * len(labels)
            )

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

class FlowerClient(
    fl.client.NumPyClient
):

    def __init__(self, cid):

        self.cid = int(cid)

        self.model = Net()

        trainset, testset = load_data()


        # =========================
        # 建立 Dirichlet Partition
        # =========================

        partitions = create_dirichlet_partition(
            trainset,
            NUM_CLIENTS,
            ALPHA
        )


        # 取得這個 Client 的資料

        client_indices = partitions[
            self.cid
        ]

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


    # =========================
    # Get Parameters
    # =========================

    def get_parameters(self, config):

        # Copy CPU arrays so training cannot mutate saved parameter snapshots.
        return [
            value.cpu().numpy().copy()
            for value
            in self.model.state_dict().values()
        ]


    # =========================
    # Set Parameters
    # =========================

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


    # =========================
    # Fit
    # =========================

    def fit(self, parameters, config):

        # Server 傳來的模型

        self.set_parameters(parameters)


        # 保存訓練前參數

        old_parameters = (
            self.get_parameters(
                config={}
            )
        )


        # =========================
        # 正常訓練
        # =========================

        train(
            self.model,
            self.trainloader
        )


        # 訓練後參數

        new_parameters = (
            self.get_parameters(
                config={}
            )
        )


        # =========================
        # Poisoning Attack
        # =========================

        if (
            ENABLE_POISON
            and self.cid == POISON_CLIENT
        ):

            print(
                f"[!] Client {self.cid} "
                f"is malicious "
                f"(scale={POISON_SCALE})"
            )

            for i in range(
                len(new_parameters)
            ):

                # 正常 Model Update

                update = (
                    new_parameters[i]
                    - old_parameters[i]
                )


                # 放大 Model Update

                new_parameters[i] = (
                    old_parameters[i]
                    + POISON_SCALE * update
                )


        return (
            new_parameters,
            len(self.trainset),
            {}
        )


    # =========================
    # Evaluate
    # =========================

    def evaluate(
        self,
        parameters,
        config
    ):

        self.set_parameters(parameters)

        loss, accuracy = test(
            self.model,
            self.testloader
        )

        return (
            float(loss),
            len(self.testset),
            {
                "accuracy":
                float(accuracy)
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
