from pathlib import Path
import numpy as np
from src.d00_utils.definitions import ROOT_DIR, PROJECT_ID, STANDARD_DATASET_FILENAME
from src.d00_utils.definitions import REL_PATHS, DATATYPE_MAP
from src.d00_utils.functions import load_wandb_data_artifact, id_from_tags
import wandb
import json
import scipy.io


def mat_to_numpy(data_x,
                 data_y,
                 num_images,
                 images_per_file,
                 datatype_key,
                 use_flag,
                 new_dataset_path,
                 # res_function=False,
                 file_count_offset=0,
                 filename_stem=None,
                 parent_dataset_id=None):

    """

    Converts image and labels arrays from 4-band SAT-6 .mat file numpy arrays to separate, panchromatic numpy arrays
    saved in separate files.

    :param data_x: image array
    :param data_y: label array (one-hot vector)
    :param num_images: number of images to include in output ('all' or int)
    :param images_per_file: number of images to include in each output numpy file
    :param datatype_key: key extracting datatype from definitions.DATATYPE_MAP (e.g. 'np.uint8': np.uint8)
    :param use_flag: 'test_vectors', 'train_vectors', 'val_vectors', 'pan'
    :param new_dataset_path: absolute directory path for output files. A subdirectory is created if
    use_flag!= test_vectors.
    :param file_count_offset: optional. If function used in a loop, file_offset_count can be used to avoid overwriting
    previous iteration's files
    :param filename_stem: optional, e.g. filename_stem 'test' will result in 'test_{i}.npz' output files. Default stem
    is use_flag
    :param parent_dataset_id: optional, allows traceability of outputs for metrics across different
    distortions
    :return: dictionary containing image/label filenames (list) and image distortion info (dict)
    """

    if not filename_stem:
        filename_stem = use_flag

    # native_res = 28
    # res = native_res  # set the default resolution to the native resolution, update with res_function
    labels = np.argmax(data_y, axis=0)
    data_type = DATATYPE_MAP[datatype_key]
    pan_images = np.asarray(np.mean(data_x, axis=2), dtype=data_type)

    res, __, num_images_original = np.shape(pan_images)

    if use_flag != 'test_vectors':
        new_data_dir = Path(new_dataset_path, REL_PATHS[use_flag])
    else:
        new_data_dir = new_dataset_path

    if not new_data_dir.is_dir():
        Path.mkdir(new_data_dir)

    image_and_label_filenames = []  # filenames for image data files AND associated label files

    if num_images == 'all':
        num_images = num_images_original
    else:
        pan_images = pan_images[:, :, :num_images]
        labels = labels[:num_images]

    if images_per_file == 'all':
        num_new_files = 1
    else:
        num_new_files = int(np.ceil(num_images / images_per_file))

    # image_distortion_info = {}
    parent_dataset_ids = {}
    if use_flag:
        parent_dataset_ids[use_flag] = parent_dataset_id

    for i in range(num_new_files):

        start_idx, end_idx = i * images_per_file, (i + 1) * images_per_file
        label_vector = labels[start_idx:end_idx]
        image_subset = pan_images[:, :, start_idx:end_idx]
        vector_length = np.shape(image_subset)[-1]  # covers remainder at end of array (last vector prob shorter)

        image_vector = np.empty((vector_length, res, res, 3), dtype=data_type)

        for idx in range(vector_length):  # a better person would make reshaping work without a loop
            new_image = image_subset[:, :, idx]
            image_vector[idx] = np.stack(3 * [new_image], axis=2)

        name_label_filename = f'{filename_stem}_{file_count_offset + i}.npz'
        np.savez_compressed(Path(new_data_dir, name_label_filename),
                            images=image_vector,
                            labels=label_vector)
        image_and_label_filenames.append(name_label_filename)

    numpy_dataset = {
        'image_and_label_filenames': image_and_label_filenames,
        'parent_dataset_ids': parent_dataset_ids
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

        parent_artifact, parent_dataset = load_wandb_data_artifact(run, parent_artifact_name, artifact_filename)

        new_dataset_id, new_dataset_rel_dir = id_from_tags(artifact_type, tags, return_dir=True)

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

            raise Exception('Expected artifact type of "train_dataset". '
                            'build_sat6_np_dataset.py should only be used to create train datasets')

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

    wandb.login()

    _description = 'build basic pan train dataset'
    _num_images = 'all'
    _images_per_file = 2048

    _config = {
        'parent_dataset_id': 'sat6_full',
        'artifact_type': 'train_dataset',
        'num_images': _num_images,
        'images_per_file': _images_per_file,
        'val_frac': 0.1,
        'datatype_key': 'np.uint8',
        'artifact_filename': STANDARD_DATASET_FILENAME,
        'description': _description,
        'tags': ['train']
    }

    _artifact = build_log_numpy(_config)
