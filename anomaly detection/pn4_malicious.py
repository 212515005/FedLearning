# Poisoning attack simulations: malicious federated client

import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomRotation, CenterCrop
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train(net, trainloader, epochs):
    print("train")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.0)
    for _ in range(epochs):
        print("epoch")
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    print("test")
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


def load_data(normal_or_fake_data):
    transform_train = Compose(
        [
            Resize(size = (256,256)),
            RandomRotation(degrees = (-3,+3)),
            CenterCrop(size=224),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

        ])
    transform_test = Compose(
        [
            Resize(size = (224,224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    # fake dataset added when communication round 15 reached
    if normal_or_fake_data == "normal_data":
        trainset = datasets.ImageFolder(root='mydata/pn_old/pn4/train', transform=transform_train)
        testset = datasets.ImageFolder(root='mydata/pneumonia/test', transform=transform_test)
    else:
        # fake dataset
        trainset = datasets.ImageFolder(root='mydata/pn4_brain/train', transform=transform_train)
        # testset = datasets.ImageFolder(root='mydata/pneumonia/test', transform=transform_test)
        # we simulate a test on the validation set of the server but we do it locally on the fake
        # client to see the impact it has on the loss and accuracy when the labels are reversed
        # (or other dataset used) from one round to another. (so we compare the local metrics of
        # the fake client with the centralized metrics of the server without the fake client (to be rerun)

    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset, batch_size=8)

net = Net().to(DEVICE)
trainloader, testloader = load_data("normal_data")
trainloader_fake_data, testloader_fake_data = load_data("fake_data")


# defintition of the client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        print("param", parameters)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("received config")
        print(config)
        # We get the server configurations to know when this client becomes "malicious"
        changing_train_loader: int = config["changing_train_loader"]
        #if changing_train_loader == "normal_data":
        if config["changing_train_loader"] == 0:
            print("normal situation")
            self.set_parameters(parameters)
            train(net, trainloader, epochs=5)
            return self.get_parameters(), len(trainloader.dataset), {}
        else:  # we train on fake data from a certain round for this client
            self.set_parameters(parameters)
            print("fake data used for training")
            train(net, trainloader_fake_data, epochs=5)
            return self.get_parameters(), len(trainloader_fake_data.dataset), {}

    def evaluate(self, parameters, config):
        if config["changing_train_loader"] == 0:
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            # return loss, len(testloader.dataset), {"accuracy": accuracy}
            return loss, len(testloader.dataset), {"accuracy": accuracy, "loss": loss}
        else:
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader_fake_data)
            # return loss, len(testloader.dataset), {"accuracy": accuracy}
            return loss, len(testloader_fake_data.dataset), {"accuracy": accuracy, "loss": loss}


# start of the client
fl.client.start_numpy_client("localhost:8080", client=FlowerClient())