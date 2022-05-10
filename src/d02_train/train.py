from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import random
import argparse
import copy
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, ROOT_DIR, PROJECT_ID, REL_PATHS
from src.d00_utils.definitions import STANDARD_CHECKPOINT_FILENAME, STANDARD_BEST_LOSS_FILENAME
from src.d00_utils.functions import load_wandb_data_artifact, load_data_vectors, load_wandb_model_artifact
from src.d00_utils.functions import id_from_tags, save_model, get_config
from src.d02_train.train_distortions import tag_to_transform
from src.d00_utils.classes import NumpyDataset
import wandb
import os

os.environ["WANDB_START_MODE"] = 'thread'

wandb.login()


# def load_data_vectors(shard_id, directory):
#     """
#     Extracts image and data vectors from the .npz file corresponding to shard_id in directory. Intended to provide
#     image/label vectors to create an instance of the NumpyDatasetBatchDistortion class.
#     """
#
#     data = load_npz_data(directory, shard_id)
#
#     image_vector = data['images']
#     label_vector = data['labels']
#
#     return image_vector, label_vector


def get_transform(distortion_tags, crop=True):
    """
    Builds transform from distortion functions brought in by tag_to_func(), which image distortion tags to image
    distortion functions.
    """

    transform_list = [transforms.ToTensor()]

    for tag in distortion_tags:
        distortion_function = tag_to_transform[tag]()
        transform_list.append(distortion_function)

    if crop:
        transform_list.extend([
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    else:
        transform_list.extend([
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
            transforms.Resize(224),
        ])

    return transforms.Compose(transform_list)


def get_shard(shard_id, directory, transform):
    """
    Creates/returns a Dataset object built form a single .npz file.
    """

    image_vector, label_vector = load_data_vectors(shard_id, directory)
    shard = NumpyDataset(image_vector,
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

    return model, running_loss, accuracy, labels, predicts


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

        model, shard_loss, shard_accuracy, __, __ = run_shard(model=model,
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
    }, step=epoch)

    print(f'Epoch {epoch} loss after ' + str(train_shard_ct).zfill(3) + f' train shards: '
                                                                        f'{train_loss:3f} (train), '
                                                                        f'{val_loss:3f} (val)')


def log_checkpoint(model, model_metadata, new_model_artifact, run):
    model_path, helper_path = save_model(model, model_metadata)
    new_model_artifact.add_file(model_path)
    new_model_artifact.add_file(helper_path)
    run.log_artifact(new_model_artifact)


def save_best_loss_model(model, model_metadata, val_losses, epoch):

    # best_loss_model_metadata = model_metadata.copy.deepcopy()
    #
    # best_loss_model_file_config = best_loss_model_metadata['model_file_config']
    # best_loss_model_rel_dir_parent = best_loss_model_file_config['model_rel_dir']
    # best_loss_model_rel_dir = str(Path(best_loss_model_rel_dir_parent, r'best_loss'))
    #
    # # best_loss_model_file_config = {
    # #     'model_rel_dir': best_loss_model_rel_dir,
    # #     'model_filename': STANDARD_BEST_LOSS_FILENAME
    # # }
    #
    # model_metadata['model_file_config'] = model_file_config  # probably redundant, but explicit update seems safer
    # model_metadata['best_loss_epoch'] = epoch
    # model_metadata['best_loss'] = val_losses[-1]
    #
    # model_path, helper_path = save_model(model, model_metadata)
    #
    # return model_path, helper_path

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

    run_tags = copy.deepcopy(config['descriptive_tags'])
    run_tags.extend(config['distortion_tags'])

    with wandb.init(project=PROJECT_ID, job_type='train_model', config=config, tags=run_tags) as run:

        config = wandb.config  # allows wandb parameter sweeps

        dataset_artifact_id = f"{config['train_dataset_id']}:{config['train_dataset_artifact_alias']}"
        starting_model_artifact_id = f"{config['starting_model_id']}:{config['starting_model_artifact_alias']}"

        __, dataset = load_wandb_data_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)

        model, arch, __ = load_wandb_model_artifact(run, starting_model_artifact_id, return_configs=True)
        config['arch'] = arch  # ensures model metadata identifies correct model architecture

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        pin_memory = config['pin_memory']
        distortion_tags = config['distortion_tags']
        crop_flag = config['crop_flag']
        transform = get_transform(distortion_tags, crop=crop_flag)
        optimizer = getattr(torch.optim, config['optimizer'], config['lr'])(model.parameters())
        loss_function = getattr(nn, config['loss_func'])()
        description = config['description']
        artifact_type = config['artifact_type']
        train_shard_ids = dataset['train']['image_and_label_filenames']
        val_shard_ids = dataset['val']['image_and_label_filenames']

        if 'num_shards' in config.keys():
            num_shards = config['num_shards']
            if num_shards != 'all':
                frac = num_shards / len(train_shard_ids)
                train_shard_ids = train_shard_ids[:num_shards]
                num_val_shards = int(frac * len(val_shard_ids))
                num_val_shards = max(1, num_val_shards)  # ensure we get at least one val shard
                val_shard_ids = val_shard_ids[:num_val_shards]

        dataset_rel_dir = dataset['dataset_rel_dir']
        train_abs_dir = Path(ROOT_DIR, dataset_rel_dir, REL_PATHS['train_vectors'])
        val_abs_dir = Path(ROOT_DIR, dataset_rel_dir, REL_PATHS['val_vectors'])

        model_id_tags = [arch]

        if 'name_string' in config.keys() and config['name_string'] is not None:
            model_id_tags.append(config['name_string'])
            new_model_id = id_from_tags(artifact_type, model_id_tags)

        else:
            model_id_tags.extend(distortion_tags)
            new_model_id = id_from_tags(artifact_type, model_id_tags)

        new_model_rel_dir = Path(REL_PATHS[artifact_type], new_model_id)

        new_model_checkpoint_file_config = {
            'model_rel_dir': str(new_model_rel_dir),
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

            artifact_metadata = dict(config)
            artifact_metadata.update({
                'epoch': epoch
            })

            new_model_artifact = wandb.Artifact(
                new_model_id,
                type=artifact_type,
                metadata=artifact_metadata,
                description=description
            )

            log_epoch_stats(train_loss, train_acc, val_loss, val_acc, train_shard_ct, epoch)
            log_checkpoint(model, dict(config), new_model_artifact, run)

            if val_losses[-1] <= min(val_losses):

                best_loss_model_metadata = copy.deepcopy(dict(config))
                best_loss_model_path, best_loss_helper_path = save_best_loss_model(model,
                                                                                   best_loss_model_metadata,
                                                                                   val_losses,
                                                                                   epoch)

        best_loss_model_artifact = wandb.Artifact(
            f'{new_model_id}_best_loss',
            type=artifact_type,
            metadata=artifact_metadata,
            description=description
        )

        best_loss_model_artifact.add_file(str(best_loss_model_path))
        best_loss_model_artifact.add_file(str(best_loss_helper_path))
        run.log_artifact(best_loss_model_artifact)
        run.name = new_model_id

    print('done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='train_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'train_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    load_tune_model(run_config)
