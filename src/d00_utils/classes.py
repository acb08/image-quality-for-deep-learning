import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset


class Sat6ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super(Sat6ResNet, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output 6 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)


class Sat6ResNet50(nn.Module):
    def __init__(self):
        super(Sat6ResNet50, self).__init__()

        self.model = models.resnet50(pretrained=True)

        # Change the output layer to output 6 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)


class Sat6DenseNet161(nn.Module):
    def __init__(self):
        super(Sat6DenseNet161, self).__init__()

        self.model = models.densenet161(pretrained=True)

        # Change the output layer to output 6 classes instead of 1000 classes
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)


class NumpyDataset(Dataset):

    def __init__(self, image_array, label_array, transform=None):
        self.label_array = label_array
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        label = self.label_array[idx]
        label = label
        if self.transform:
            image = self.transform(image)
        return image, label
