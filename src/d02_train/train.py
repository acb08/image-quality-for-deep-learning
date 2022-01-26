from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import random
import os
import argparse
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, ROOT_DIR, PROJECT_ID, REL_PATHS
from src.d00_utils.functions import load_wandb_artifact, load_npz_data
from src.d01_pre_processing.distortions import tag_to_func
import wandb

wandb.login()


class NumpyDatasetBatchDistortion(Dataset):

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


def train_on_shard(model, shard, batch_size, num_workers, shuffle, pin_memory, device, loss_func, optimizer):
    """
    Train model using a single numpy image vector
    """

    train_loader = get_loader(shard,
                              batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              pin_memory=pin_memory)

    for i, (images, labels) in enumerate(train_loader):

        model.zero_grad()

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_func(labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    return 'loss', 'accuracy'


def eval_on_shard():
    """
    Validate model using a single numpy image vector
    """
    return 'loss', 'accuracy'


def load_data_vectors(shard_id, directory):

    data = load_npz_data(directory, shard_id)

    image_vector = data['images']
    label_vector = data['labels']

    return image_vector, label_vector


def get_transform(distortion_tags):

    transform_list = [transforms.ToTensor()]

    for tag in distortion_tags:
        distortion_function = tag_to_func(tag)
        transform_list.append(distortion_function)

    transform_list.extend([
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    return transforms.Compose(transform_list)


def get_shard(shard_id, directory, transform):

    image_vector, label_vector = load_data_vectors(shard_id, directory)
    shard = NumpyDatasetBatchDistortion(image_vector,
                                        label_vector,
                                        transform)

    return shard


def get_loader(shard, batch_size, shuffle, pin_memory, num_workers):

    loader = DataLoader(shard,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=pin_memory,
                        num_workers=num_workers)

    return loader


def get_mean_accuracy(scaled_accuracies, total_samples):
    """
    Calculated mean accuracy when each value in scaled accuracies represents an average accuracy multiplied
    by the its underlying sample count, where total samples is the sum of the sample counts for each value in
    scaled_accuracies
    """
    scaled_accuracies = np.asarray(scaled_accuracies) / total_samples
    mean_accuracy = float(np.mean(scaled_accuracies))
    return mean_accuracy


def train(model, train_shard_ids, transform):

    random.shuffle(train_shard_ids)

    running_loss = 0
    total_samples = 0
    scaled_accuracies = []

    for shard_id in train_shard_ids:

        train_shard, shard_length = get_shard(shard_id, transform)
        train_loader = get_loader(train_shard, 'batch_size')
        loss, accuracy = train_on_shard(model, train_loader)

        total_samples += shard_length
        running_loss += loss / shard_length

        scaled_accuracy = shard_length * accuracy
        scaled_accuracies.append(scaled_accuracy)

    mean_accuracy = get_mean_accuracy(scaled_accuracies, total_samples)

    return running_loss, mean_accuracy


def evaluate(model, eval_shard_ids, transform):

    running_loss = 0
    total_samples = 0
    scaled_accuracies = []

    for shard_id in eval_shard_ids:

        eval_shard, shard_length = get_shard(shard_id, transform)
        eval_loader = get_loader(eval_shard, 'batch_size')
        loss, accuracy = eval_on_shard(model, eval_loader)

        total_samples += shard_length
        running_loss += loss / shard_length

        # scale accuracy by number of samples to get appropriately weighted mean accuracy later
        scaled_accuracy = shard_length * accuracy
        scaled_accuracies.append(scaled_accuracy)

    mean_accuracy = get_mean_accuracy(scaled_accuracies, total_samples)

    return running_loss, mean_accuracy


if __name__ == '__main__':

    _manual_config = True

    if _manual_config:

        _dataset_id = '0026_numpy'
        _artifact_alias = 'latest'

        config = {
            'dataset_id': _dataset_id,
            'artifact_alias': _artifact_alias
        }

    _artifact_id = f'{_dataset_id}:{_artifact_alias}'

    with wandb.init(project=PROJECT_ID, job_type='transfer_dataset') as run:

        _artifact, _dataset = load_wandb_artifact(run, _artifact_id, STANDARD_DATASET_FILENAME)

        _train_shard_ids = _dataset['train']['image_and_label_filenames']
        _val_shard_ids = _dataset['val']['image_and_label_filenames']

    _dataset_rel_dir = _dataset['dataset_rel_dir']
    _train_rel_dir = Path(_dataset_rel_dir, REL_PATHS['train_vectors'])
    _val_rel_dir = Path(_dataset_rel_dir, REL_PATHS['val_vectors'])
    _train_abs_dir = Path(ROOT_DIR, _train_rel_dir)
    _val_abs_dir = Path(ROOT_DIR, _val_rel_dir)

    _transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # verify ability to load vectors from shard ids (done)
    # verify ability to generate shards (really mini datasets) (done)
    for _shard_id in _train_shard_ids:
        # _train_images, _train_labels = load_data_vectors(_shard_id, _train_abs_dir)
        _train_shard = get_shard(_shard_id, _train_abs_dir, _transform)

    for _shard_id in _val_shard_ids:
        _val_images, _val_labels = load_data_vectors(_shard_id, _val_abs_dir)
        print(_val_labels)
        _val_shard = get_shard(_shard_id, _val_abs_dir, _transform)
    #

    # verify ability to create dataloader from shards (done)
    _train_loader = get_loader(_train_shard, 5, shuffle=True, num_workers=0, pin_memory=False)
    _val_loader = get_loader(_val_shard, 6, shuffle=False, num_workers=0, pin_memory=False)

    for i, batch in enumerate(_train_loader):

        print(type(batch))
