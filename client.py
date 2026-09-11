import sys

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================
# Poisoning 設定
# =========================

# 哪一個 Client 是惡意 Client
POISON_CLIENT = 0

# Poisoning 強度
POISON_SCALE = 10


# =========================
# 1. Model
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
# 2. Dataset
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
# 3. Train
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

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()


# =========================
# 4. Test
# =========================

def test(model, testloader):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():

        for images, labels in testloader:

            output = model(images)

            loss = criterion(output, labels)

            loss_total += loss.item()

            _, predicted = torch.max(output, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return loss_total / len(testloader), accuracy


# =========================
# 5. Flower Client
# =========================

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, cid):

        self.cid = cid

        trainset, testset = load_data()

        # =========================
        # Non-IID Data Partition
        # =========================

        client_labels = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
            4: [8, 9]
        }

        my_labels = client_labels[int(cid)]

        indices = [
            i
            for i, label in enumerate(trainset.targets)
            if int(label) in my_labels
        ]

        self.trainset = torch.utils.data.Subset(
            trainset,
            indices
        )

        self.testset = testset

        self.trainloader = DataLoader(
            self.trainset,
            batch_size=32,
            shuffle=True
        )

        self.testloader = DataLoader(
            self.testset,
            batch_size=128
        )

        self.model = Net()


    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]


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


    def fit(self, parameters, config):

        self.set_parameters(parameters)

        # =========================
        # 保存 Server 傳下來的 Global Model
        # =========================

        old_parameters = self.get_parameters(config={})

        # =========================
        # 本地訓練
        # =========================

        train(self.model, self.trainloader)

        # 本地訓練後的 Model
        new_parameters = self.get_parameters(config={})


        # =========================
        # Model Poisoning
        # =========================

        if int(self.cid) == POISON_CLIENT:

            print(
                f"[!] Client {self.cid} is malicious "
                f"(scale={POISON_SCALE})"
            )

            for i in range(len(new_parameters)):

                # 計算 Local Update
                update = (
                    new_parameters[i]
                    - old_parameters[i]
                )

                # 放大 Update
                new_parameters[i] = (
                    old_parameters[i]
                    + POISON_SCALE * update
                )


        return (
            new_parameters,
            len(self.trainset),
            {}
        )


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
# 6. Start Client
# =========================

if __name__ == "__main__":

    cid = sys.argv[1]

    client = FlowerClient(cid)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

