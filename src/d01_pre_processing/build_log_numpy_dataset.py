from pathlib import Path
import numpy as np
from PIL import Image
from src.d00_utils.definitions import ROOT_DIR, PROJECT_ID, STANDARD_DATASET_FILENAME
from src.d00_utils.definitions import REL_PATHS, DATATYPE_MAP
from src.d00_utils.functions import log_metadata, load_wandb_artifact, name_from_tags
import wandb
import json
import random

wandb.login()


def build_data_vector(image_dir, names_labels, image_shape, datatype_key):

    datatype = DATATYPE_MAP[datatype_key]

    height, width, channels = image_shape
    num_images_to_load = len(names_labels)
    image_vector_shape = (num_images_to_load, height, width, channels)
    image_vector = np.zeros(image_vector_shape, dtype=datatype)
    label_vector = np.zeros(num_images_to_load, dtype=np.int32)

    for idx, (image_name, label) in enumerate(names_labels):

        image_path = Path(image_dir, image_name)
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=datatype)
        image_vector[idx] = image
        label_vector[idx] = label

    return image_vector, label_vector


def train_val_split(all_names_labels, val_frac, val_shuffle=False):

    """
    :param all_names_labels: list [(image_name, image_label), ...]. Note: image name can also contain
    multiple elements of a relative filepath
    :param val_frac: fraction of images to carve aside for validation
    :param val_shuffle: if True, val split shuffled randomly
    :return: train_slice, val_slice, both lists of the form [(image_name, image_label), ...]. The train_slice
    is randomly shuffled to ensure a stochastic mix of images in each train image vector. The validation
    slice preserves
    """

    start_idx = 0
    stop_idx = len(all_names_labels)
    step = int(1 / val_frac)

    indices = list(np.arange(stop_idx))

    val_split_indices = indices[start_idx:stop_idx:step]
    train_split_indices = set(indices) - set(val_split_indices)

    train_split = [all_names_labels[idx] for idx in train_split_indices]
    val_split = [all_names_labels[idx] for idx in val_split_indices]

    random.shuffle(train_split)  # avoid having all one class in each train file

    if val_shuffle:
        random.shuffle(val_split)

    return train_split, val_split


def transfer_to_numpy(parent_names_labels,
                      starting_img_parent_rel_dir,
                      num_images,
                      images_per_file,
                      image_shape,
                      data_type,
                      train_val_path_flag,
                      new_dataset_path):

    """
    :param parent_names_labels: list of name label pairs
    :param starting_img_parent_rel_dir: path to image directory
    :param num_images: total number of images to incorporate (int or keyword 'all')
    :param images_per_file: int, number of images to include in each vector
    :param image_shape: list [h, w, c] (because json unable to handle tuples)
    :param data_type: numpy data type (e.g. np.uint8)
    :param train_val_path_flag: str, 'train_vectors' or 'val_vectors'
    :param new_dataset_path: absolute path to parent directory for new files
    :return: dict with filenames for .npz vector files and keys to access the files image vectors and label vectors
    """

    starting_img_rel_dir = Path(starting_img_parent_rel_dir, REL_PATHS['images'])
    starting_img_dir = Path(ROOT_DIR, starting_img_rel_dir)

    new_data_dir = Path(new_dataset_path, REL_PATHS[train_val_path_flag])
    if not new_data_dir.is_dir():
        Path.mkdir(new_data_dir)

    image_and_label_filenames = []  # filenames for image data files AND associated label files

    if num_images == 'all':
        num_images = len(parent_names_labels)
    num_new_files = int(np.ceil(num_images / images_per_file))

    for i in range(num_new_files):

        start_idx, end_idx = i * images_per_file, (i + 1) * images_per_file
        names_labels_subset = parent_names_labels[start_idx:end_idx]

        image_vector, label_vector = build_data_vector(starting_img_dir,
                                                       names_labels_subset,
                                                       image_shape,
                                                       data_type)

        name_label_filename = f'{train_val_path_flag}_{i}.npz'
        np.savez_compressed(Path(new_data_dir, name_label_filename),
                            images=image_vector,
                            labels=label_vector)
        image_and_label_filenames.append(name_label_filename)

    numpy_dataset = {
        'image_and_label_filenames': image_and_label_filenames,
    }

    return numpy_dataset


def build_log_numpy(config):
    """
    transfers dataset's images to into numpy files, records dataset in .json file, and logs dataset on W&B
    """

    parent_dataset_id = config['parent_dataset_id']
    artifact_type = config['artifact_type']
    num_images = config['num_images']
    images_per_file = config['images_per_file']
    image_shape = config['image_shape']
    datatype_key = config['datatype_key']
    artifact_filename = config['artifact_filename']
    val_frac = config['val_frac']
    tags = config['tags']
    val_shuffle = config['val_shuffle']

    parent_artifact_name = f'{parent_dataset_id}:latest'

    with wandb.init(project=PROJECT_ID, job_type='transfer_dataset') as run:

        parent_artifact, parent_dataset = load_wandb_artifact(run, parent_artifact_name, artifact_filename)

        new_dataset_id = name_from_tags(artifact_type, tags)
        new_dataset_rel_parent_dir = REL_PATHS[artifact_type]
        new_dataset_rel_dir = Path(new_dataset_rel_parent_dir, new_dataset_id)

        new_dataset_abs_path = Path(ROOT_DIR, new_dataset_rel_dir)
        Path.mkdir(new_dataset_abs_path)
        full_dataset_path = Path(new_dataset_abs_path, artifact_filename)

        parent_names_labels = parent_dataset['names_labels']

        train_split, val_split = train_val_split(parent_names_labels, val_frac, val_shuffle=val_shuffle)

        starting_img_parent_rel_dir = parent_dataset['dataset_rel_dir']

        train_path_flag = 'train_vectors'
        val_path_flag = 'val_vectors'

        dataset_train_split = transfer_to_numpy(train_split,
                                                starting_img_parent_rel_dir,
                                                num_images,
                                                images_per_file,
                                                image_shape,
                                                datatype_key,
                                                train_path_flag,
                                                new_dataset_abs_path)

        dataset_val_split = transfer_to_numpy(val_split,
                                              starting_img_parent_rel_dir,
                                              num_images,
                                              images_per_file,
                                              image_shape,
                                              datatype_key,
                                              val_path_flag,
                                              new_dataset_abs_path)

        new_dataset = {
            'train': dataset_train_split,
            'val': dataset_val_split
        }

        run_metadata = config
        run_metadata_additions = {
            'artifact_filename': artifact_filename,
            'root_dir_at_run': str(ROOT_DIR),
            'dataset_rel_dir': str(new_dataset_rel_dir)
        }
        run_metadata.update(run_metadata_additions)

        log_metadata(artifact_type, new_dataset_id, run_metadata)
        new_dataset.update(run_metadata)

        artifact = wandb.Artifact(new_dataset_id,
                                  type=artifact_type,
                                  metadata=run_metadata)

        with open(full_dataset_path, 'w') as file:
            json.dump(new_dataset, file)

        artifact.add_file(full_dataset_path)
        run.log_artifact(artifact)
        run.name = new_dataset_id
        wandb.finish()

    return parent_artifact


if __name__ == "__main__":

    _description = 'test of saving dataset in numpy files'
    _num_images = 500
    _images_per_file = 30

    _config = {
        'parent_dataset_id': 'train_256_standard',
        'artifact_type': 'train_dataset',
        'num_images': _num_images,
        'images_per_file': _images_per_file,
        'val_frac': 0.05,
        'image_shape': [256, 256, 3],
        'datatype_key': 'np.uint8',
        'artifact_filename': STANDARD_DATASET_FILENAME,
        'description': _description,
        'tags': ['numpy'],
        'val_shuffle': False
    }

    _artifact = build_log_numpy(_config)
