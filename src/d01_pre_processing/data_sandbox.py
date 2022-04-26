import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from src.d00_utils.functions import load_data_vectors
import matplotlib.pyplot as plt


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
        if self.transform:
            image = self.transform(image)
        return image, label


def get_original(data_array, idx):
    original = data_array[idx]
    return original


def to_tensor_homebrew(array):
    h, w, c = np.shape(array)
    new_array = np.reshape(array, (c, h, w))
    new_array = new_array / 255
    new_array = torch.tensor(new_array)
    return new_array


# num_samples = 10
# labels = np.arange(num_samples)
# shape = (num_samples, 256, 256, 1)
# data = np.random.randint(0, 255, shape, dtype=np.uint8)
# tensor = torch.tensor(data)
#
# _transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# dataset = NumpyDataset(data, labels, transform=_transform)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# mean_diffs = []
#
# for (images, labels) in loader:
#
#     print(np.shape(images), labels)
#
#     for i, label in enumerate(labels):
#         original = get_original(data, int(label))
#         original_torchified = to_tensor_homebrew(original)
#         diff = images[i] - original_torchified
#         mean_diffs.append(float(torch.mean(diff)))
#
#
# def foo(a, b):
#
#     print(a)
#     print(b)
#
#     return
#
#
# _bar = {
#     'a': 1,
#     'b': 2,
# }
#
# foo(**_bar)

def view_npz_file_images(how_many, directory, filename):

    image_vector, label_vector = load_data_vectors(filename, directory)
    for i in range(how_many):
        image = image_vector[i]
        plt.figure()
        plt.imshow(image)
        plt.title(str(label_vector[i]))
        plt.show()


_directory = '/home/acb6595/places/datasets/test/0006_tst_r_scan_pl/1-res'
_filenames = 'val_vectors_31.npz'

_filenames = [f'test_{i}.npz' for i in range(31)]

for _filename in _filenames:

    view_npz_file_images(1, _directory, _filename)

