from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import random
import os
import argparse
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, ROOT_DIR, PROJECT_ID, REL_PATHS
from src.d00_utils.functions import load_wandb_dataset_artifact, load_npz_data, load_wandb_artifact_model
from src.d00_utils.functions import id_from_tags, get_config, read_json_artifact, string_from_tags
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
        label = label
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data_vectors(shard_id, directory):
    """
    Extracts image and data vectors from the .npz file corresponding to shard_id in directory. Intended to provide
    image/label vectors to create an instance of the NumpyDatasetBatchDistortion class.
    """

    data = load_npz_data(directory, shard_id)

    image_vector = data['images']
    label_vector = data['labels']

    return image_vector, label_vector


def get_transform(distortion_tags):
    """
    Builds transform from distortion functions brought in by tag_to_func(), which image distortion tags to image
    distortion functions.
    """

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
    """
    Creates/returns a Dataset object built form a single .npz file.
    """

    image_vector, label_vector = load_data_vectors(shard_id, directory)
    shard = NumpyDatasetBatchDistortion(image_vector,
                                        label_vector,
                                        transform)

    shard_length = len(label_vector)

    return shard, shard_length


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
    mean_accuracy = np.sum(scaled_accuracies) / total_samples
    mean_accuracy = float(mean_accuracy)
    return mean_accuracy


def get_raw_accuracy(predicts, targets):
    """
    :param predicts: list or numpy array
    :param targets: list or numpy array
    :return: float
    """

    predicts = np.asarray(predicts)
    targets = np.asarray(targets)
    accuracy = float(np.mean(np.equal(predicts, targets)))

    return accuracy


# def eval_on_shard(model, shard, batch_size, num_workers, shuffle, pin_memory, device, loss_func):
#     """
#     Validate model using a single numpy image vector
#     """
#
#     loader = get_loader(shard,
#                         batch_size,
#                         num_workers=num_workers,
#                         shuffle=shuffle,
#                         pin_memory=pin_memory)
#
#     running_loss = 0
#     predicts = []
#     labels = []
#
#     model.eval()
#
#     for i, (images, targets) in enumerate(loader):
#         images, targets = images.to(device), targets.to(device).long()
#         outputs = model(images)
#         loss = loss_func(outputs, targets)
#
#         predicted_classes = list(torch.max(outputs, 1))[1].cpu()
#         predicts.extend(predicted_classes)
#         labels.extend(list(targets.cpu()))
#
#         running_loss += loss.item() / len(targets)
#
#     accuracy = get_raw_accuracy(predicts, labels)
#
#     return running_loss, accuracy
#
#
# def train_on_shard(model, shard, batch_size, num_workers, shuffle, pin_memory, device, loss_func, optimizer):
#     """
#     Train model on a single shard, where a shard is a Dataset object build from a .npz file consisting of a numpy
#     image vector and its associated numpy label vector.
#     """
#
#     loader = get_loader(shard,
#                         batch_size,
#                         num_workers=num_workers,
#                         shuffle=shuffle,
#                         pin_memory=pin_memory)
#
#     running_loss = 0
#     predicts = []
#     labels = []
#
#     model.train()
#
#     for i, (images, targets) in enumerate(loader):
#         model.zero_grad()
#
#         images, targets = images.to(device), targets.to(device).long()
#         outputs = model(images)
#         loss = loss_func(outputs, targets)
#
#         predicted_classes = list(torch.max(outputs, 1))[1].cpu()
#         predicts.extend(predicted_classes)
#         labels.extend(list(targets.cpu()))
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() / len(targets)
#
#     accuracy = get_raw_accuracy(predicts, labels)
#
#     return running_loss, accuracy


def run_shard(model, train_eval_flag, shard, batch_size, num_workers, pin_memory, device, loss_func, optimizer):
    """
    Train OR evaluate (determined by train_eval_flag) a model on a single shard, where a shard is a Dataset object build from
    a .npz file consisting of a numpy image vector and its associated numpy label vector.

    :param model: PyTorch model
    :param train_eval_flag: str, 'train' or 'eval' determines whether function runs in training or evaluation mode
    :param shard: Dataset object to be run
    :param batch_size: int
    :param num_workers: int
    :param pin_memory: int
    :param device: str
    :param loss_func: PyTorch module
    :param optimizer: PyTorch module
    :return: model, loss, accuracy
    """

    if train_eval_flag == 'train':
        shuffle = True
        model.train()

    elif train_eval_flag == 'eval':
        shuffle = False
        model.eval()
    else:
        raise Exception('invalid train_eval_flag')

    loader = get_loader(shard,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory)

    running_loss = 0
    predicts = []
    labels = []

    for i, (images, targets) in enumerate(loader):

        model.zero_grad()

        images, targets = images.to(device), targets.to(device).long()
        outputs = model(images)
        loss = loss_func(outputs, targets)

        predicted_classes = list(torch.max(outputs, 1))[1].cpu()
        predicts.extend(predicted_classes)
        labels.extend(list(targets.cpu()))

        if train_eval_flag == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() / len(targets)

    accuracy = get_raw_accuracy(predicts, labels)

    return model, running_loss, accuracy


def propagate(model,
              train_eval_flag,
              shard_ids,
              data_abs_dir,
              transform,
              batch_size,
              num_workers,
              pin_memory,
              device,
              loss_func,
              optimizer):
    if train_eval_flag == 'train':
        random.shuffle(shard_ids)
        model.train()
    elif train_eval_flag == 'eval':
        model.eval()
    else:
        raise Exception('invalid train_eval_flag')

    running_loss = 0
    total_samples = 0
    weighted_accuracies = []
    shard_lengths = []
    shard_accuracies = []

    for shard_id in shard_ids:
        shard, shard_length = get_shard(shard_id, data_abs_dir, transform)

        model, shard_loss, shard_accuracy = run_shard(model=model,
                                                      train_eval_flag=train_eval_flag,
                                                      shard=shard,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers,
                                                      pin_memory=pin_memory,
                                                      device=device,
                                                      loss_func=loss_func,
                                                      optimizer=optimizer)

        shard_accuracies.append(shard_accuracy)
        shard_lengths.append(shard_length)

        total_samples += shard_length

        scaled_loss = shard_loss / shard_length
        running_loss += scaled_loss
        weighted_accuracy = shard_length * shard_accuracy
        weighted_accuracies.append(weighted_accuracy)

    mean_accuracy = get_mean_accuracy(weighted_accuracies, total_samples)

    return running_loss, mean_accuracy, shard_accuracies, shard_lengths


# def epoch_eval(model,
#                shard_ids,
#                eval_abs_dir,
#                transform,
#                batch_size,
#                num_workers,
#                pin_memory,
#                device,
#                loss_func):
#     running_loss = 0
#     total_samples = 0
#     scaled_accuracies = []
#
#     for shard_id in shard_ids:
#         eval_shard, shard_length = get_shard(shard_id, eval_abs_dir, transform)
#         eval_loader = get_loader(eval_shard,
#                                  batch_size=batch_size,
#                                  shuffle=False,
#                                  pin_memory=pin_memory,
#                                  num_workers=num_workers)
#         loss, accuracy = eval_on_shard(model,
#                                        eval_loader)
#
#         total_samples += shard_length
#         running_loss += loss / shard_length
#
#         # scale accuracy by number of samples to get appropriately weighted mean accuracy later
#         scaled_accuracy = shard_length * accuracy
#         scaled_accuracies.append(scaled_accuracy)
#
#     mean_accuracy = get_mean_accuracy(scaled_accuracies, total_samples)
#
#     return running_loss, mean_accuracy


def load_tune_model():
    return


if __name__ == '__main__':

    global model

    _manual_config = True

    if _manual_config:
        _dataset_id = '0023_numpy'
        _dataset_artifact_alias = 'latest'
        _model_id = 'resnet18_pretrained'
        _model_artifact_alias = 'latest'
        _distortion_tags = ['pan_c', 'r4', 'b4', 'n4']
        _batch_size = 32
        _num_workers = 0
        _shuffle = True
        _pin_memory = True
        _optimizer = 'Adam'
        _loss_func = 'CrossEntropyLoss'
        _description = 'dry run of pipeline to train/log models'
        _artifact_type = 'model'

        _config = {
            'dataset_id': _dataset_id,
            'dataset_artifact_alias': _dataset_artifact_alias,
            'model_id': _model_id,
            'model_artifact_alias': _model_artifact_alias,
            'distortion_tags': _distortion_tags,
            'batch_size': _batch_size,
            'num_workers': _num_workers,
            'shuffle': _shuffle,
            'pin_memory': _pin_memory,
            'optimizer': _optimizer,
            'loss_func': _loss_func,
            'description': _description,
            'artifact_type': _artifact_type
        }

    with wandb.init(project=PROJECT_ID, job_type='train_model', config=_config) as run:

        config = wandb.config

        dataset_artifact_id = f'{config.dataset_id}:{config.dataset_artifact_alias}'
        model_artifact_id = f'{config.model_id}:{config.model_artifact_alias}'

        __, dataset = load_wandb_dataset_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)

        model = load_wandb_artifact_model(run, model_artifact_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        batch_size = config.batch_size
        num_workers = config.num_workers
        shuffle = config.shuffle
        pin_memory = config.pin_memory
        distortion_tags = config.distortion_tags
        transform = get_transform(distortion_tags)
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
        loss_function = getattr(nn, config.loss_func)()
        description = config.description
        artifact_type = config.artifact_type

        train_shard_ids = dataset['train']['image_and_label_filenames']
        val_shard_ids = dataset['val']['image_and_label_filenames']
        dataset_rel_dir = dataset['dataset_rel_dir']
        train_abs_dir = Path(ROOT_DIR, dataset_rel_dir, REL_PATHS['train_vectors'])
        val_abs_dir = Path(ROOT_DIR, dataset_rel_dir, REL_PATHS['val_vectors'])

        new_model_id = id_from_tags(artifact_type, distortion_tags)
        new_model_rel_parent_dir = REL_PATHS[artifact_type]
        new_model_abs_dir = Path(ROOT_DIR, new_model_rel_parent_dir, new_model_id)
        Path.mkdir(new_model_abs_dir)

        train_losses = []
        train_accuracies = []

        train_loss, train_acc, shard_accuracies, shard_lengths = propagate(model,
                                                                           'train',
                                                                           train_shard_ids,
                                                                           train_abs_dir,
                                                                           transform,
                                                                           batch_size,
                                                                           num_workers,
                                                                           pin_memory,
                                                                           device,
                                                                           loss_function,
                                                                           optimizer)

        val_loss, val_acc, val_shard_accs, val_shard_lengths = propagate(model,
                                                                         'eval',
                                                                         val_shard_ids,
                                                                         val_abs_dir,
                                                                         transform,
                                                                         batch_size,
                                                                         num_workers,
                                                                         pin_memory,
                                                                         device,
                                                                         loss_function,
                                                                         optimizer)

        # TODO: integrate helper file (helper.json) method for keeping model loading from being hideous
        # helper file corollary: log best_loss, best_acc model every epoch, basically just following the
        # approach used in original CIS-linux version of the code. On second thought, log the model every every
        # epoch with the basic filename of model_<tag1>_<tag2>_etc.pt. In parallel, store best loss/best acc (probably
        # just best loss) model(s) and log them at the end of training. This approach carries some redundancy, but it's
        # probably the best bet for ensuring I don't lose anything...
        # TODO: log trained model as new wandb artifact
        # TODO: bundle train/val loop in single function
        # TODO: finalize whether to make model (plus possibly loss_function and optimizer global variables)
