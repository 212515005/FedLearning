from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader, XYDataset

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='emnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/emnist/mydata/pneumonia',
                                      )
        self.num_classes = 2
        self.save_data = self.XYData_to_json
        self.selected_labels = [i for i in range(2)]

    def load_data(self):
        transform_train = transforms.Compose(
            [
                transforms.Resize(size = (256,256)),
                transforms.RandomRotation(degrees = (-2,+2)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                #transforms.Normalize([0.5179, 0.5179, 0.5179],[0.1967, 0.1967, 0.1967])
            ])
        transform_test = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                #transforms.Normalize([0.5179, 0.5179, 0.5179],[0.1967, 0.1967, 0.1967])
            ])



        lb_convert = {}
        for i in range(len(self.selected_labels)):
            lb_convert[self.selected_labels[i]] = i
        #self.train_data = datasets.EMNIST(self.rawdata_path, split='letters', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.train_data = datasets.ImageFolder(root='./benchmark/emnist/mydata/pneumonia/train', transform=transform_train)
        #self.test_data = datasets.EMNIST(self.rawdata_path, split='letters', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.ImageFolder(root='./benchmark/emnist/mydata/pneumonia/test', transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=10,
                                                   shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=10,
                                                  shuffle=False)

        classes = self.train_data.classes
        print(classes)

        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # get some random training images
        #dataiter = iter(self.train_loader)
        #images, labels = dataiter.next()

        # show images
        #imshow(torchvision.utils.make_grid(images))

        print("start_train -1")
        train_didxs = [did for did in range(len(self.train_data)) if self.train_data[did][1] in self.selected_labels]
        print("start_train")
        train_data_x = [self.train_data[did][0].tolist() for did in train_didxs]
        print("start_train2")
        train_data_y = [lb_convert[self.train_data[did][1]] for did in train_didxs]
        print("start_train3")
        self.train_data = XYDataset(train_data_x, train_data_y)
        print("finito")
        test_didxs = [did for did in range(len(self.test_data)) if self.test_data[did][1] in self.selected_labels]
        test_data_x = [self.test_data[did][0].tolist() for did in test_didxs]
        test_data_y = [lb_convert[self.test_data[did][1]] for did in test_didxs]
        self.test_data = {'x':test_data_x, 'y':test_data_y}
        print("load_data finished")

    def convert_data_for_saving(self):
        train_x, train_y = self.train_data.tolist()
        self.train_data = {'x':train_x, 'y':train_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)



'''
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader, XYDataset

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='emnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/emnist/mydata/pneumonia',
                                      )
        self.num_classes = 2
        self.save_data = self.XYData_to_json
        self.selected_labels = [i for i in range(2)]

    def load_data(self):
        # dataset has PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1]
        transform_train = transforms.Compose(
            [
                transforms.Resize(size = (256,256)),
                transforms.RandomRotation(degrees = (-2,+2)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                #transforms.Normalize([0.1433, 0.2760, 0.4969],[0.8589, 0.8781, 0.8742])))
                transforms.Normalize([0.5179, 0.5179, 0.5179],[0.1967, 0.1967, 0.1967])
                #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
            ])
        transform_test = transforms.Compose(
            [
                transforms.Resize(size = (224,224)),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])



        lb_convert = {}
        for i in range(len(self.selected_labels)):
            lb_convert[self.selected_labels[i]] = i
        #self.train_data = datasets.EMNIST(self.rawdata_path, split='letters', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.train_data = datasets.ImageFolder(root='./benchmark/emnist/mydata/pneumonia/train', transform=transform_train)
        #self.test_data = datasets.EMNIST(self.rawdata_path, split='letters', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.ImageFolder(root='./benchmark/emnist/mydata/pneumonia/test', transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64,
                                                   shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=16,
                                                  shuffle=False)

        classes = self.train_data.classes
        print(classes)

        def imshow(img):
            #img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        def get_mean_and_std(dataloader):
            channels_sum, channels_squared_sum, num_batches = 0, 0, 0
            for data, _ in dataloader:
                # Mean over batch, height and width, but not over the channels
                channels_sum += torch.mean(data, dim=[0, 2, 3])
                channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
                num_batches += 1

            mean = channels_sum / num_batches

            # std = sqrt(E[X^2] - (E[X])^2)
            std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
            print("mean", mean)
            print(std)

            return mean, std

        #get_mean_and_std(self.train_loader)
        #mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
        #print(yolo[0].mean([0, 2, 3]), yolo[0].std([0, 2, 3]))
        # get some random training images
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))

        print("start_train -1")
        train_didxs = [did for did in range(len(self.train_data)) if self.train_data[did][1] in self.selected_labels]
        print("start_train")
        train_data_x = [self.train_data[did][0].tolist() for did in train_didxs]
        print("start_train2")
        train_data_y = [lb_convert[self.train_data[did][1]] for did in train_didxs]
        print("start_train3")
        self.train_data = XYDataset(train_data_x, train_data_y)
        print("finito")
        test_didxs = [did for did in range(len(self.test_data)) if self.test_data[did][1] in self.selected_labels]
        test_data_x = [self.test_data[did][0].tolist() for did in test_didxs]
        test_data_y = [lb_convert[self.test_data[did][1]] for did in test_didxs]
        self.test_data = {'x':test_data_x, 'y':test_data_y}
        print("load_data finished")

    def convert_data_for_saving(self):
        train_x, train_y = self.train_data.tolist()
        self.train_data = {'x':train_x, 'y':train_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
'''
