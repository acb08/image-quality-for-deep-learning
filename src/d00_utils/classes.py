import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


class VariablePoissonNoiseChannelReplicated(object):
    """
    Selects Poisson noise standard deviation on sigma_poisson_range with uniform probability
    inclusive of the endpoints.

    generates zero-centered poisson noise with mean selected as described above, divides the noise by 255,
    adds to input tensor and clamps the output to fall on [0, 1]

    Copies the noise across input image channels so that a panchromatic image where the RGB channels are identical
    has the same noise added to each channel.

    """

    def __init__(self, sigma_poisson_range, clamp=True):
        self.low = sigma_poisson_range[0] ** 2
        self.high = sigma_poisson_range[1] ** 2 + 1  # add one because torch.randint excludes right endpoint
        self.clamp = clamp

    def __call__(self, tensor):
        # batch_size, channels, height, width = tensor.shape
        channels, height, width = tensor.shape
        noise_shape = (1, height, width)

        lambda_poisson = torch.randint(self.low, self.high, (1,))
        noise = (torch.poisson(torch.ones(noise_shape) * lambda_poisson).float() - lambda_poisson) / 255
        # channel_replicated_noise = noise.repeat(1, channels, 1, 1)
        channel_replicated_noise = noise.repeat(channels, 1, 1)
        noisy_tensor = tensor + channel_replicated_noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class VariableImageResize(object):
    """
    Builds a transform bank to re-size images and provides a method for randomly selecting the size of the
    transform to be used before being called.
    """

    def __init__(self, sizes, interpolation_mode='bilinear', antialias=False):
        self.sizes = sizes
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.transform_bank = self.build_transform_bank()
        self.size_keys = list(self.transform_bank.keys())

    def build_transform_bank(self):
        transform_bank = {}
        for size in self.sizes:
            size = int(size)
            if self.interpolation_mode == 'bilinear':
                new_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                                       antialias=self.antialias),
                     transforms.ToPILImage()]
                )
            elif self.interpolation_mode == 'bicubic':
                new_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=self.antialias),
                     transforms.ToPILImage()]
                )
            else:
                raise Exception('Invalid interpolation_mode')
            transform_bank[size] = new_transform
        print('build_transform_bank called')
        return transform_bank

    def get_size_key(self):
        """
        randomly returns one of the class's size keys, which can then be used in the __call__() method. Allows the user
        to know the resolution ahead of time for things like initializing arrays of the correct shape.
        """
        size_key = random.choice(self.size_keys)
        return size_key

    def __call__(self, image, size_key, dtype=np.uint8):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        transform_use = self.transform_bank[size_key]
        image = transform_use(image)
        image = np.asarray(image, dtype=dtype)
        return image


class VariableResolution(object):

    """
    Randomly selects a resize transform per the sizes argument passed in the __init__() method. Intended for use
    in a dataloader
    """

    def __init__(self, sizes, interpolation_mode='bilinear', antialias=False):
        self.sizes = sizes
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.transform_bank = self.build_transform_bank()

    def build_transform_bank(self):
        transform_bank = []
        for size in self.sizes:
            size = int(size)
            if self.interpolation_mode == 'bilinear':
                new_transform = transforms.Resize(size,
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=self.antialias)
            elif self.interpolation_mode == 'bicubic':
                new_transform = transforms.Resize(size,
                                                  interpolation=transforms.InterpolationMode.BICUBIC,
                                                  antialias=self.antialias)
            else:
                raise Exception('Invalid interpolation_mode')
            transform_bank.append(new_transform)
        print('build_transform_bank called')
        return transform_bank

    def __call__(self, tensor):
        transform_use = random.choice(self.transform_bank)
        return transform_use(tensor)
