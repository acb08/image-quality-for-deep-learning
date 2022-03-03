from pathlib import Path
import numpy as np
from PIL import Image
from src.d00_utils.definitions import ROOT_DIR, PROJECT_ID, STANDARD_DATASET_FILENAME
from src.d00_utils.definitions import REL_PATHS, DATATYPE_MAP
from src.d00_utils.functions import load_wandb_dataset_artifact, id_from_tags
import wandb
import json
import random
import scipy.io

wandb.login()


def mat_to_numpy(data_x,
                 data_y,
                 num_images,
                 images_per_file,
                 datatype_key,
                 train_val_path_flag,
                 new_dataset_path):

    labels = np.argmax(data_y, axis=0)
    data_type = DATATYPE_MAP[datatype_key]
    images = np.asarray(np.mean(data_x, axis=2), dtype=data_type)

    if train_val_path_flag == 'train_vectors' or train_val_path_flag == 'val_vectors':
        new_data_dir = Path(new_dataset_path, REL_PATHS[train_val_path_flag])
    else:
        new_data_dir = new_dataset_path

    if not new_data_dir.is_dir():
        Path.mkdir(new_data_dir)

    image_and_label_filenames = []  # filenames for image data files AND associated label files

    if num_images == 'all':
        num_images = np.shape(data_x)[-1]

    if images_per_file == 'all':
        num_new_files = 1
    else:
        num_new_files = int(np.ceil(num_images / images_per_file))

    for i in range(num_new_files):

        start_idx, end_idx = i * images_per_file, (i + 1) * images_per_file
        label_vector = labels[start_idx:end_idx]
        image_subset = images[:, :, start_idx:end_idx]
        vector_length = np.shape(image_subset)[-1]  # covers remainder at end of array (last vector prob shorter)
        image_vector = np.empty((vector_length, 28, 28, 3), dtype=data_type)
        for idx in range(vector_length):  # a better person would make reshaping work without a loop
            image_vector[idx] = np.stack(3*[image_subset[:, :, idx]], axis=2)
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
    datatype_key = config['datatype_key']
    artifact_filename = config['artifact_filename']
    val_frac = config['val_frac']
    tags = config['tags']
    description = config['description']

    parent_artifact_name = f'{parent_dataset_id}:latest'

    with wandb.init(project=PROJECT_ID, job_type='transfer_dataset', tags=tags, notes=description,
                    config=config) as run:

        parent_artifact, parent_dataset = load_wandb_dataset_artifact(run, parent_artifact_name, artifact_filename)

        new_dataset_id = id_from_tags(artifact_type, tags)

        new_dataset_rel_parent_dir = REL_PATHS[artifact_type]
        new_dataset_rel_dir = Path(new_dataset_rel_parent_dir, new_dataset_id)
        new_dataset_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir)
        Path.mkdir(new_dataset_abs_dir)
        full_dataset_path = Path(new_dataset_abs_dir, artifact_filename)

        data_abs_path = Path(ROOT_DIR, parent_dataset['dataset_rel_dir'], parent_dataset['dataset_filename'])
        data = scipy.io.loadmat(data_abs_path)

        if artifact_type == 'train_dataset':

            data_x, data_y = data['train_x'], data['train_y']
            num_images_total = np.shape(data_x)[-1]
            num_val = int(val_frac * num_images_total)
            num_train = num_images_total - num_val

            train_split_x, train_split_y = data_x[:, :, :, :num_train], data_y[:, :num_train]
            val_split_x, val_split_y = data_x[:, :, :, num_train:], data_y[:, num_train:]

            train_path_flag = 'train_vectors'
            val_path_flag = 'val_vectors'

            dataset_train_split = mat_to_numpy(train_split_x,
                                               train_split_y,
                                               num_images,
                                               images_per_file,
                                               datatype_key,
                                               train_path_flag,
                                               new_dataset_abs_dir)

            dataset_val_split = mat_to_numpy(val_split_x,
                                             val_split_y,
                                             num_images,
                                             images_per_file,
                                             datatype_key,
                                             val_path_flag,
                                             new_dataset_abs_dir)

            new_dataset = {
                'train': dataset_train_split,
                'val': dataset_val_split
            }

        else:

            data_x, data_y = data['test_x'], data['test_y']
            test_path_flag = 'test_vectors'

            dataset_test = mat_to_numpy(data_x,
                                        data_y,
                                        num_images,
                                        images_per_file,
                                        datatype_key,
                                        test_path_flag,
                                        new_dataset_abs_dir)

            new_dataset = {
                'test': dataset_test
            }

        run_metadata = config
        run_metadata_additions = {
            'artifact_filename': artifact_filename,
            'root_dir_at_run': str(ROOT_DIR),
            'dataset_rel_dir': str(new_dataset_rel_dir)
        }
        run_metadata.update(run_metadata_additions)
        new_dataset.update(run_metadata)

        artifact = wandb.Artifact(new_dataset_id,
                                  type=artifact_type,
                                  metadata=run_metadata)

        with open(full_dataset_path, 'w') as file:
            json.dump(new_dataset, file)

        artifact.add_file(full_dataset_path)
        artifact.metadata = run_metadata
        run.log_artifact(artifact)
        run.name = new_dataset_id
        wandb.finish()

    return parent_artifact


if __name__ == "__main__":

    _description = 'save undistorted test dataset with new artifact type tagging'
    _num_images = 5000
    _images_per_file = 1000

    _config = {
        'parent_dataset_id': 'sat6_full',
        'artifact_type': 'test_dataset',
        'num_images': _num_images,
        'images_per_file': _images_per_file,
        'val_frac': None,
        'datatype_key': 'np.uint8',
        'artifact_filename': STANDARD_DATASET_FILENAME,
        'description': _description,
        'tags': ['micro']
    }

    _artifact = build_log_numpy(_config)
