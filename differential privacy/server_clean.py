# Differential Privacy: federated server

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomRotation, CenterCrop
from tqdm import tqdm
from torchvision import datasets
from typing import List, Tuple
from flwr.common import Metrics
from typing import Dict, Optional, Tuple
from collections import OrderedDict


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


def get_eval_fn(model: torch.nn.Module, toy: bool):
    if toy:
        transform_train = Compose(
            [
                Resize(size=(256, 256)),
                RandomRotation(degrees=(-3, +3)),
                CenterCrop(size=224),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        transform_test = Compose(
            [
                Resize(size=(224, 224)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        valset = datasets.ImageFolder(root='mydata/pneumonia/test', transform=transform_test)

    valLoader = DataLoader(valset, batch_size=8)
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, valLoader)
        print("server evaluation")
        print("evaluate du server (eval_fn):", loss, {"accuracy": accuracy})

        with open('zcentralized_metrics.csv', 'a') as fd:
            fd.write(str(accuracy) + ',')  # accuracy centralized at the end of each round
        with open('zcentralized_losses.csv', 'a') as fd:
            fd.write(str(loss) + ',')  # loss centralized at the end of each round

        return loss, {"accuracy": accuracy}

    return evaluate


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_a = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.batch_a = nn.BatchNorm2d(num_features=12)
        self.relu_a = nn.ReLU()
        self.max_a = nn.MaxPool2d(kernel_size=2)
        self.conv_b = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu_b = nn.ReLU()
        self.conv_c = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_b = nn.BatchNorm2d(num_features=32)
        self.relu_c = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=2)

    def forward(self, input):
        output = self.conv_a(input)
        output = self.batch_a(output)
        output = self.relu_a(output)
        output = self.max_a(output)
        output = self.conv_b(output)
        output = self.relu_b(output)
        output = self.conv_c(output)
        output = self.batch_b(output)
        output = self.relu_c(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)

        return output


net = Net().to(DEVICE)
# trainloader, testloader = load_data()
model = net


strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    fraction_fit=1.0,
    min_available_clients=5,
    min_fit_clients=5,
    fraction_eval=1.0,
    min_eval_clients=5,
    accept_failures=False,

    # new fn
    eval_fn=get_eval_fn(model, True),
)


# start of the serverr
fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 30},
    strategy=strategy,
)
