#!/usr/bin/env python
import json

from src.d00_utils.functions import load_wandb_dataset_artifact, id_from_tags, get_config, \
    log_config
from src.d00_utils.definitions import REL_PATHS, DATATYPE_MAP, STANDARD_DATASET_FILENAME, PROJECT_ID, ROOT_DIR
from src.d01_pre_processing.build_sat6_np_dataset import mat_to_numpy
from src.d01_pre_processing.distortions import tag_to_image_distortion
from src.d00_utils.functions import load_data_vectors
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import scipy.io
import wandb

wandb.login()


def distort_np_img_vec(vector_in, distortion_function, datatype_key, distortion_type_flag):

    native_resolution = 28
    vector_length, res, __, num_channels = np.shape(vector_in)

    if distortion_type_flag == 'res':
        res = distortion_function.get_size_key()  # distortion function can be an instance of a class also

    datatype = DATATYPE_MAP[datatype_key]
    vector_out = np.empty((vector_length, res, res, num_channels), dtype=datatype)

    distortion_values = []

    for i in range(vector_length):

        if distortion_type_flag == 'res':
            distorted_image = distortion_function(vector_in[i], res, dtype=datatype)
            distortion_value = res / native_resolution
        else:
            starting_img = Image.fromarray(vector_in[i])
            distorted_image, __, distortion_value = distortion_function(starting_img)
            distorted_image = np.asarray(distorted_image, dtype=datatype)

        vector_out[i] = distorted_image
        distortion_values.append(distortion_value)

    return vector_out, distortion_values


def distort_dataset(dataset, new_dataset_path, distortion_tag, distortion_type_flag, datatype_key, dataset_split_key):

    image_and_label_filenames = dataset[dataset_split_key]['image_and_label_filenames']  # constant parent to child
    image_distortion_info = dataset[dataset_split_key]['image_distortion_info']  # inherits and updates parent to child
    last_distortion_type_flag = dataset['last_distortion_type_flag']

    parent_image_file_dir = Path(new_dataset_path, REL_PATHS[last_distortion_type_flag])

    new_image_file_dir = Path(new_dataset_path, REL_PATHS[distortion_type_flag])
    Path.mkdir(new_image_file_dir)

    if distortion_type_flag == 'res':
        distortion_function = tag_to_image_distortion[distortion_tag]()
    else:
        distortion_function = tag_to_image_distortion[distortion_tag]

    for name_label_filename in image_and_label_filenames:

        image_vector, label_vector = load_data_vectors(name_label_filename, parent_image_file_dir)
        distorted_image_vector, distortion_values = distort_np_img_vec(image_vector,
                                                                       distortion_function,
                                                                       datatype_key,
                                                                       distortion_type_flag)
        np.savez_compressed(Path(new_image_file_dir, name_label_filename),
                            images=distorted_image_vector,
                            labels=label_vector)

        if name_label_filename in image_distortion_info:
            image_distortion_info[name_label_filename][distortion_type_flag] = distortion_values
        else:
            image_distortion_info[name_label_filename] = {
                distortion_type_flag: distortion_values
            }

    dataset['last_distortion_type_flag'] = distortion_type_flag

    return dataset


def distort_log_numpy(config):

    parent_dataset_id = config['parent_dataset_id']
    parent_artifact_alias = config['parent_artifact_alias']
    parent_artifact_filename = config['parent_artifact_filename']
    if parent_artifact_filename == 'standard':
        parent_artifact_filename = STANDARD_DATASET_FILENAME
    artifact_type = config['artifact_type']
    num_images = config['num_images']
    images_per_file = config['images_per_file']
    datatype_key = config['datatype_key']
    artifact_filename = config['artifact_filename']
    if artifact_filename == 'standard':
        artifact_filename = STANDARD_DATASET_FILENAME
    distortion_tags = config['distortion_tags']
    distortion_type_flags = config['distortion_type_flags']
    dataset_split_key = config['dataset_split_key']
    iterations = config['iterations']
    description = config['description']

    with wandb.init(project=PROJECT_ID, job_type='distort_dataset', tags=distortion_tags, notes=description,
                    config=config) as run:

        parent_artifact_name = f'{parent_dataset_id}:{parent_artifact_alias}'
        parent_artifact, parent_dataset = load_wandb_dataset_artifact(run,
                                                                      parent_artifact_name,
                                                                      parent_artifact_filename)

        new_dataset_id, new_dataset_rel_dir = id_from_tags(artifact_type, distortion_tags, return_dir=True)
        new_dataset_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir)
        Path.mkdir(new_dataset_abs_dir)

        data_abs_path = Path(ROOT_DIR, parent_dataset['dataset_rel_dir'], parent_dataset['dataset_filename'])
        data = scipy.io.loadmat(data_abs_path)
        data_x, data_y = data['test_x'], data['test_y']

        run_metadata = config
        run_metadata_additions = {
            'artifact_filename': artifact_filename,
            'root_dir_at_run': str(ROOT_DIR),
            'dataset_rel_dir': str(new_dataset_rel_dir)
        }
        run_metadata.update(run_metadata_additions)

        pan_image_label_filenames = []
        image_distortion_info = {}

        for i in range(iterations):

            file_count_offset = len(pan_image_label_filenames)
            new_data_subset = mat_to_numpy(data_x,
                                           data_y,
                                           num_images,
                                           images_per_file,
                                           datatype_key,
                                           # distortion_type_flags[0],
                                           'pan',
                                           new_dataset_abs_dir,
                                           # res_distortion_func,
                                           file_count_offset=file_count_offset,
                                           filename_stem='test',
                                           parent_dataset_id=parent_artifact_name)

            subset_image_label_filenames = new_data_subset['image_and_label_filenames']
            pan_image_label_filenames.extend(subset_image_label_filenames)
            parent_dataset_ids = new_data_subset['parent_dataset_ids']
            # image_distortion_info.update(res_data_subset['image_distortion_info'])

        pan_dataset = {dataset_split_key: {
            'image_and_label_filenames': pan_image_label_filenames,
            'image_distortion_info': image_distortion_info},
            'last_distortion_type_flag': 'pan',  # enables subsequent dataset users to find correct subdirectory
            'parent_dataset_ids': parent_dataset_ids
        }

        pan_dataset.update(run_metadata)
        pan_artifact_id = f"{new_dataset_id}_pan"
        pan_artifact = wandb.Artifact(pan_artifact_id,
                                      type=artifact_type,
                                      metadata=run_metadata)
        pan_artifact_path = Path(new_dataset_abs_dir, REL_PATHS['pan'], artifact_filename)
        with open(pan_artifact_path, 'w') as file:
            json.dump(pan_dataset, file)

        pan_artifact.add_file(pan_artifact_path)
        pan_artifact.metadata = run_metadata
        run.log_artifact(pan_artifact)
        pan_artifact.wait()  # allow artifact upload to finish before its used downstream

        # with the new pan artifact created, loop through distortion tags and create a new artifact for
        # the output of each associated distortion function, where each subsequent artifact has the last artifact
        # for its parent
        parent_dataset_id = pan_artifact_id
        parent_artifact_name = f'{pan_artifact_id}:latest'

        for i, distortion_tag in enumerate(distortion_tags):

            distortion_type_flag = distortion_type_flags[i]
            parent_artifact, parent_dataset = load_wandb_dataset_artifact(run,
                                                                          parent_artifact_name,
                                                                          artifact_filename)

            new_dataset = distort_dataset(parent_dataset,
                                          new_dataset_abs_dir,
                                          distortion_tag,
                                          distortion_type_flag,
                                          datatype_key,
                                          dataset_split_key)

            new_dataset['parent_dataset_ids'][distortion_type_flag] = parent_dataset_id
            new_artifact_id = f"{new_dataset_id}_{distortion_type_flags[i]}"
            new_artifact = wandb.Artifact(new_artifact_id,
                                          type=artifact_type,
                                          metadata=run_metadata)
            new_artifact_path = Path(new_dataset_abs_dir, REL_PATHS[distortion_type_flag], artifact_filename)
            with open(new_artifact_path, 'w') as file:
                json.dump(new_dataset, file)

            new_artifact.add_file(new_artifact_path)
            new_artifact.metadata = run_metadata
            run.log_artifact(new_artifact)
            new_artifact.wait()

            # now update the parent artifact id and name for the next iteration of the loop
            parent_dataset_id = new_artifact_id
            parent_artifact_name = f'{new_artifact_id}:latest'

        run.name = new_dataset_id
        log_config(new_dataset_abs_dir, dict(config), return_path=False)

    return pan_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    print(run_config)

    pan_test_dataset = distort_log_numpy(run_config)
