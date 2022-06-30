import torchvision.datasets
from torchvision import datasets, transforms
from benchmark.toolkits import ClassificationCalculator, DefaultTaskGen, IDXTaskReader

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='cifar10_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/mine/pneumonia',
                                      )
        self.num_classes = 2
        self.save_data = self.IDXData_to_json
        self.visualize = self.visualize_by_class
        self.datasrc = {
            'class_path': 'torchvision.datasets',
            'class_name': 'ImageFolder',
            'train_args': {
                'root': '"./benchmark/mine/pneumonia/train"',
                'transform': 'transforms.Compose([transforms.ToTensor()])'  # je feed les données déjà préprocess ici pour gagner du temps mais on peut appliquer le préprocess ici aussi
            },
            'test_args': {
                'root': '"./benchmark/mine/pneumonia/test"',
                'transform': 'transforms.Compose([transforms.ToTensor()])' # pareil
            }
        }


    def load_data(self):
        self.train_data = datasets.ImageFolder(root='./benchmark/mine/pneumonia/train',
                                               transform=transforms.Compose([transforms.ToTensor()])) # pareil
        self.test_data = datasets.ImageFolder(root='./benchmark/mine/pneumonia/test',
                                              transform=transforms.Compose([transforms.ToTensor()])) # pareil


class TaskReader(IDXTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassificationCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
