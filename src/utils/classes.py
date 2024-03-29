from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# from src.utils.functions import _time_string
from src.utils import detection_functions
from src.utils import definitions


# from src.utils.coco_functions import xywh_to_xyxy


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


class PseudoArgs:
    """
    Imitates argparse.ArgumentParser() object to allow calls of functions.get_config(args) without actually using
    argparse (i.e. it's useful when a function needs to retrieve a config file)
    """
    def __init__(self, config_dir, config_name):
        self.config_dir = config_dir
        self.config_name = config_name


class COCO(Dataset):

    def __init__(self, image_directory, instances,
                 transform=transforms.Compose([transforms.ToTensor()]),
                 cutoff=None,
                 yolo_fmt=False):

        self.image_directory = Path(image_directory)
        if not self.image_directory.is_absolute():
            self.image_directory = Path(definitions.ROOT_DIR, self.image_directory)
        self.instances = instances
        self.images = self.instances['images']
        if cutoff is not None:
            self.images = self.images[:cutoff]
        self.annotations = self.instances['annotations']
        self.image_ids = detection_functions.get_image_ids(self.images)
        self.mapped_boxes_labels = detection_functions.map_boxes_labels(self.annotations, self.image_ids)
        self.transform = transform
        self.yolo_fmt = yolo_fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_data = self.images[idx]
        file_name = image_data['file_name']
        image_id = image_data['id']
        image = Image.open(Path(self.image_directory, file_name)).convert('RGB')
        image_annotations = self.mapped_boxes_labels[image_id]

        if len(image_annotations['boxes']) == 0:
            image_annotations = detection_functions.background_annotation(image)

        image_annotations['image_id'] = torch.tensor(image_id)

        if not self.yolo_fmt:
            image = self.transform(image)

        return image, image_annotations


class Illustrator(COCO):

    """
    Returns PIL images and bounding boxes in list format for use in illustrating images, instead of returning torch
    tensors for training/testing models
    """

    def __init__(self, image_directory, instances, cutoff=None):
        super().__init__(image_directory, instances, cutoff, yolo_fmt=False)
        self._mapped_licenses_urls = None

    def __getitem__(self, idx):
        image_data = self.images[idx]
        file_name = image_data['file_name']
        image_id = image_data['id']
        image = Image.open(Path(self.image_directory, file_name)).convert('RGB')
        image_annotations = self.mapped_boxes_labels[image_id]

        if len(image_annotations['boxes']) == 0:
            image_annotations = detection_functions.background_annotation(image)

        image_annotations = {key: item.tolist() for key, item in image_annotations.items()}

        image_annotations['image_id'] = image_id

        return image, image_annotations, file_name

    def license_url_map(self):
        if self._mapped_licenses_urls is None:
            mapped_licenses = {image['file_name']: [image['license'], image['flickr_url']] for image in self.images}
            self._mapped_licenses_urls = mapped_licenses
        return self._mapped_licenses_urls
