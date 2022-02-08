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
from src.d00_utils.definitions import STANDARD_CHECKPOINT_FILENAME, STANDARD_BEST_LOSS_FILENAME
from src.d00_utils.functions import load_wandb_dataset_artifact, load_npz_data, load_wandb_artifact_model
from src.d00_utils.functions import id_from_tags, save_model, get_config, read_json_artifact, string_from_tags
from src.d01_pre_processing.distortions import tag_to_func
import wandb
import json

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
              optimizer,
              epoch,
              run):
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

    return model, running_loss, mean_accuracy, shard_accuracies, shard_lengths


def log_epoch_stats(train_loss, train_acc, val_loss, val_acc, train_shard_ct, epoch):

    wandb.log({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'epoch': epoch,
    }, step=train_shard_ct)

    print('Loss after ' + str(train_shard_ct).zfill(3) + f' train shards: '
                                                         f'{train_loss:3f} (train), '
                                                         f'{val_loss:3f} (val)')


def log_checkpoint(model, model_metadata, epoch, new_model_artifact, run):

    model_path, helper_path = save_model(model, model_metadata)
    new_model_artifact.add_file(model_path)
    new_model_artifact.add_file(helper_path)
    run.log_artifact(new_model_artifact)


def save_best_loss_model(model, model_metadata, val_losses, epoch):

    model_file_config = model_metadata['model_file_config']
    model_rel_dir = model_file_config['model_rel_dir']
    best_loss_model_rel_dir = str(Path(model_rel_dir, r'best_loss'))

    model_file_config['model_rel_dir'] = best_loss_model_rel_dir
    model_file_config['model_filename'] = STANDARD_BEST_LOSS_FILENAME

    model_metadata['model_file_config'] = model_file_config  # probably redundant, but explicit update seems safer
    model_metadata['best_loss_epoch'] = epoch
    model_metadata['best_loss'] = val_losses[-1]

    model_path, helper_path = save_model(model, model_metadata)

    return model_path, helper_path


def load_tune_model(config):

    with wandb.init(project=PROJECT_ID, job_type='train_model', config=config) as run:

        config = wandb.config

        dataset_artifact_id = f'{config.train_dataset_id}:{config.train_dataset_artifact_alias}'
        starting_model_artifact_id = f'{config.starting_model_id}:{config.starting_model_artifact_alias}'

        __, dataset = load_wandb_dataset_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)

        model = load_wandb_artifact_model(run, starting_model_artifact_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        num_epochs = config.num_epochs
        batch_size = config.batch_size
        num_workers = config.num_workers
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
        # new_model_rel_parent_dir = REL_PATHS[artifact_type]
        new_model_rel_dir = Path(REL_PATHS[artifact_type], new_model_id)
        # new_model_abs_dir = Path(ROOT_DIR, new_model_rel_parent_dir, new_model_rel_dir)
        # Path.mkdir(new_model_abs_dir)

        new_model_checkpoint_file_config = {
            'model_rel_dir': new_model_rel_dir,
            'model_filename': STANDARD_CHECKPOINT_FILENAME
        }
        config['model_file_config'] = new_model_checkpoint_file_config

        # log all relevant model versions within the artifact created here

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        train_shard_ct = 0

        for epoch in range(num_epochs):

            model, train_loss, train_acc, train_shard_accuracies, shard_lengths = propagate(model,
                                                                                            'train',
                                                                                            train_shard_ids,
                                                                                            train_abs_dir,
                                                                                            transform,
                                                                                            batch_size,
                                                                                            num_workers,
                                                                                            pin_memory,
                                                                                            device,
                                                                                            loss_function,
                                                                                            optimizer,
                                                                                            epoch,
                                                                                            run)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_shard_ct += len(train_shard_accuracies)

            model, val_loss, val_acc, val_shard_accuracies, val_shard_lengths = propagate(model,
                                                                                          'eval',
                                                                                          val_shard_ids,
                                                                                          val_abs_dir,
                                                                                          transform,
                                                                                          batch_size,
                                                                                          num_workers,
                                                                                          pin_memory,
                                                                                          device,
                                                                                          loss_function,
                                                                                          optimizer,
                                                                                          epoch,
                                                                                          run)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            new_model_artifact = wandb.Artifact(
                new_model_id,
                type=artifact_type,
                metadata=dict(config),
                description=description
            )

            log_epoch_stats(train_loss, train_acc, val_loss, val_acc, train_shard_ct, epoch)
            log_checkpoint(model, dict(config), epoch, new_model_artifact, run)

            if val_losses[-1] <= min(val_losses):

                best_loss_model_path, best_loss_helper_path = save_best_loss_model(model,
                                                                                   dict(config),
                                                                                   val_losses,
                                                                                   epoch)

    best_loss_model_artifact = wandb.Artifact(
        f'{new_model_id}_best_loss',
        type=artifact_type,
        metadata=dict(config),
        description=description
    )
    best_loss_model_artifact.add_file(str(best_loss_model_path))
    best_loss_model_artifact.add_file(str(best_loss_helper_path))

    print('done')


if __name__ == '__main__':

    # global model

    _manual_config = True

    if _manual_config:
        _train_dataset_id = '0023_numpy'
        _train_dataset_artifact_alias = 'latest'
        _starting_model_id = 'resnet18_pretrained'
        _starting_model_artifact_alias = 'latest'
        _distortion_tags = ['pan_c', 'r4', 'b4', 'n4']
        _num_epochs = 2
        _batch_size = 32
        _num_workers = 0
        _pin_memory = True
        _optimizer = 'Adam'
        _loss_func = 'CrossEntropyLoss'
        _description = 'dry run of pipeline to train/log models'
        _artifact_type = 'model'

        _config = {
            'train_dataset_id': _train_dataset_id,
            'train_dataset_artifact_alias': _train_dataset_artifact_alias,
            'starting_model_id': _starting_model_id,
            'starting_model_artifact_alias': _starting_model_artifact_alias,
            'distortion_tags': _distortion_tags,
            'num_epochs': _num_epochs,
            'batch_size': _batch_size,
            'num_workers': _num_workers,
            'pin_memory': _pin_memory,
            'optimizer': _optimizer,
            'loss_func': _loss_func,
            'description': _description,
            'artifact_type': _artifact_type
        }

        load_tune_model(_config)

