# Poisoning attack simulations: federated server

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


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# local analysis
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # local restults
    print(metrics)
    with open('zdistributed_metrics.csv', 'a') as fd:
        fd.write(str(sum(accuracies) / sum(examples))+',') # metrics_distributed accuracy at each round (moyenné)
    with open('zdistributed_losses.csv', 'a') as fd:
        fd.write(str(sum(losses) / sum(examples))+',') # losses_distributed at each round (moyenné)

    with open('zdistributed_losses_each_client.csv', 'a') as fd:
        losses_each = [m["loss"] for num_examples, m in metrics]
        fd.write(str(losses_each) +',') # losses_distributed at each round (of each client)
    with open('zdistributed_metrics_each_client.csv', 'a') as fd:
        accuracies_each = [m["accuracy"] for num_examples, m in metrics]
        fd.write(str(accuracies_each) + ',') # metrics_distributed accuracy at each round (of each client)

    return {"accuracy": sum(accuracies) / sum(examples)}


# validation of the global model on testing set
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


# server side evaluation (evaluation function)
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
    # called at each round for evaluation
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
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


# new config to manually simulate attacks at round 15
# on évalue finalement en ne laissant que le client modifié
def fit_config(rnd: int):
    if rnd < 15:
        changing_train_loader = 0
    else:
        changing_train_loader = 1
    return {"changing_train_loader": changing_train_loader, "round": rnd}


def evaluate_config(rnd: int):
    if rnd < 15:
        changing_train_loader = 0
    else:
        changing_train_loader = 1
    return {"changing_train_loader": changing_train_loader}


strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    fraction_fit=1.0,
    min_available_clients=1,
    min_fit_clients=1,
    fraction_eval=1.0,
    min_eval_clients=1,
    accept_failures=False,

    # new fn
    eval_fn=get_eval_fn(model, True),

    # new config
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
)


# start of the server
fl.server.start_server(
    server_address="localhost:8080",
    config={"num_rounds": 30},
    strategy=strategy,
)
