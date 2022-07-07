#!/usr/bin/env python
import json
from src.d00_utils.functions import load_wandb_data_artifact, get_config
from src.d00_utils.definitions import REL_PATHS, WANDB_PID, ROOT_DIR, STANDARD_ENTROPY_PROPERTIES_FILENAME, \
    STANDARD_DATASET_FILENAME, DISTORTION_TYPES, STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME
from src.d00_utils.functions import load_data_vectors
from src.d04_analysis.entropy_functions import tag_to_entropy_function
import numpy as np
import argparse
from pathlib import Path
import wandb
from PIL import Image, ImageOps

wandb.login()

IMAGE_MODE = 'L'  # currently, adding random noise across all channels (i.e. noise between channels is independent).
# I need to do one of the following: (a) modify noise distortion functions to add identical noise to all three
# image channels (b) modify entropy functions to account for noise independence across channels


def get_entropy_artifact_name(dataset_id, effective):
    """
    Returns standard artifact name for a dataset's entropy properties. Allows an entropy artifact to be used along with
    the dataset from which it is derived without manually specifying the entropy artifact name. This automatic naming
    approach is a departure from the goal of making all links between artifacts explicit and avoiding reliance on
    formatted names to link data. Because entropy artifacts are created after distortion datasets, this approach avoids
    having to modify the dataset artifact or create a third artifact formally linking the two.

    :param dataset_id: str, name of wandb dataset artifact
    :param effective: bool, specifies whether entropy artifact captures effective entropy
    :return: str, standard entropy artifact name
    """

    if ':' in dataset_id:
        dataset_id = dataset_id.split(':')[0]
    entropy_artifact_name = f"{dataset_id}_entropy"
    if effective:
        entropy_artifact_name = get_effective_entropy_equiv(entropy_artifact_name)
    return entropy_artifact_name


def get_effective_entropy_equiv(entropy_artifact_name):
    return f'{entropy_artifact_name}_effective'


def get_vec_entropy_vals(vector_in, entropy_functions):

    """
    Extracts entropy an entropy value for each function in entropy_functions for each image in vector_in. Returns a
    dictionary of the following form:
        {entropy_type_key_0: [entropy_val_0, ..., entropy_val_n]
        entropy_type_key_1: [entropy_val_1, ..., entropy_val_n]
        ...}
    where n is given by np.shape(vector_in)[0] and corresponds to the number of images in vector_n
    """

    vector_length, res, __, num_channels = np.shape(vector_in)
    entropy_values = {}

    for i in range(vector_length):

        image = vector_in[i]
        if IMAGE_MODE == 'L':
            image = Image.fromarray(image)
            image = ImageOps.grayscale(image)
        else:
            image = image[:, :, 0]

        for j, entropy_func in enumerate(entropy_functions):
            key, entropy_val = entropy_func(image)
            if i == 0:
                entropy_values[key] = [entropy_val]
            else:
                entropy_values[key].append(entropy_val)

    return entropy_values


def get_entropy_properties(dataset, dataset_path, entropy_functions, dataset_split_key):

    """

    :param dataset: dictionary containing image vector filenames and distortion type flags denoting the
    last distortion type applied
    :param dataset_path: absolute path to the umbrella directory containing a subdirectory for each distortion applied
    :param entropy_functions: list containing entropy functions to be used for extracting entropy properties
    :param dataset_split_key: should be 'test'. Used to maintain parallel between train and test datasets, where
    train dataset contain 'train' and 'val' splits
    :return: entropy properties dict of the following form:
        {shard_id_0: {
        entropy_type_key_0: [entropy_value_0, entropy_value_1, ...],
        entropy_type_key_1: [entropy_value_0, entropy_value_1, ...],
        ...}
        shard_id_1: {
        entropy_type_key_0: [entropy_value_0, entropy_value_1, ...],
        entropy_type_key_1: [entropy_value_0, entropy_value_1, ...],
        ...}
        }
    """

    image_and_label_filenames = dataset[dataset_split_key]['image_and_label_filenames']  # constant parent to child
    last_distortion_type_flag = dataset['last_distortion_type_flag']

    parent_image_file_dir = Path(dataset_path, REL_PATHS[last_distortion_type_flag])

    entropy_properties = {}

    for i, name_label_filename in enumerate(image_and_label_filenames):

        image_vector, __ = load_data_vectors(name_label_filename, parent_image_file_dir)
        entropy_values = get_vec_entropy_vals(image_vector, entropy_functions)

        if name_label_filename in entropy_properties:
            entropy_properties[name_label_filename].update(entropy_values)
        else:
            entropy_properties[name_label_filename] = entropy_values

        if i % 10 == 0 and i > 0:
            print(f'Measured {name_label_filename}, {i + 1} / {len(image_and_label_filenames)} image vector files')

    return entropy_properties


def measure_dataset_entropy_properties(config):

    """
    Measures the entropy properties of a dataset logged as a W&B artifact and logs these entropy properties as new
    W&B artifacts.
    """

    dataset_id = config['starting_dataset_id']
    dataset_artifact_alias = 'latest'
    num_analysis_stages = config['num_analysis_stages']
    parent_artifact_filename = STANDARD_DATASET_FILENAME
    dataset_split_key = config['dataset_split_key']

    new_artifact_type = 'entropy_properties'
    entropy_function_tags = config['entropy_function_tags']
    entropy_functions = [tag_to_entropy_function[tag] for tag in entropy_function_tags]
    description = config['description']
    measure_effective_entropy = config['measure_effective_entropy']
    overwrite_previous = config['overwrite_previous']

    if measure_effective_entropy:
        noise_dataset_id = dataset_id
        if noise_dataset_id[-5:] != 'noise':
            raise Exception("Expected dataset_id ending in noise")

    measured_dataset_directories = []

    with wandb.init(project=WANDB_PID, job_type='measure_entropy_properties', notes=description,
                    config=config) as run:

        for i in range(num_analysis_stages):

            parent_artifact_name = f'{dataset_id}:{dataset_artifact_alias}'
            parent_artifact, dataset = load_wandb_data_artifact(run, parent_artifact_name, parent_artifact_filename)

            distortion_type_flag = dataset['last_distortion_type_flag']
            dataset_abs_dir = Path(ROOT_DIR, dataset['dataset_rel_dir'])  # umbrella directory w/ all distortion stages

            shard_entropy_properties = get_entropy_properties(dataset, dataset_abs_dir, entropy_functions,
                                                              dataset_split_key)
            print(f'Completed {dataset_id} entropy measurement, function tags: {entropy_function_tags}')

            dataset_distortion_stage_abs_dir = Path(dataset_abs_dir, REL_PATHS[distortion_type_flag])
            measured_dataset_directories.append(str(dataset_distortion_stage_abs_dir))

            entropy_properties_path = Path(dataset_distortion_stage_abs_dir, STANDARD_ENTROPY_PROPERTIES_FILENAME)

            if entropy_properties_path.is_file() and not overwrite_previous:
                with open(entropy_properties_path, 'r') as file:
                    dataset_entropy_properties = json.load(file)
                dataset_entropy_properties['shard_entropy_properties'] = shard_entropy_properties
            else:
                dataset_entropy_properties = {'shard_entropy_properties': shard_entropy_properties}

            dataset_entropy_properties.update(config)
            dataset_entropy_properties['dataset_id'] = dataset_id

            with open(entropy_properties_path, 'w') as file:
                json.dump(dataset_entropy_properties, file)

            new_artifact_id = get_entropy_artifact_name(dataset_id, effective=False) #f"{dataset_id}_entropy"
            new_artifact = wandb.Artifact(new_artifact_id,
                                          type=new_artifact_type,
                                          metadata=config)

            new_artifact.add_file(entropy_properties_path)
            run.log_artifact(new_artifact)
            new_artifact.wait()

            current_iteration_distortion_type = DISTORTION_TYPES[-(i + 1)]  # move from end of list child to parent
            dataset_id = dataset['parent_dataset_ids'][current_iteration_distortion_type]

        if measure_effective_entropy:

            shard_effective_entropy_properties = get_effective_entropy_values(measured_dataset_directories)
            effective_entropy_properties = {'shard_effective_entropy_properties': shard_effective_entropy_properties}
            effective_entropy_properties.update(config)
            effective_entropy_properties['dataset_id'] = noise_dataset_id
            effective_entropy_properties_path = Path(measured_dataset_directories[0],
                                                     STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME)

            with open(effective_entropy_properties_path, 'w') as file:
                json.dump(effective_entropy_properties, file)

            effective_entropy_artifact_id = get_entropy_artifact_name(noise_dataset_id, effective=True)
            effective_entropy_artifact = wandb.Artifact(
                effective_entropy_artifact_id,
                type='effective_entropy_properties',
                metadata=config
            )
            effective_entropy_artifact.add_file(effective_entropy_properties_path)
            run.log_artifact(effective_entropy_artifact)

        run.name = f'{noise_dataset_id}_entropy'

        # return measured_dataset_directories


def get_effective_entropy_values(measured_dataset_directories, ignore_directory_name_warnings=False):

    """
    Extracts effective entropy values for using the entropy_properties.json files in measured_dataset_directories.

    :param measured_dataset_directories: directories containing dataset entropy properties, in the format
    [noise_directory, blur_directory, ...].
    :param ignore_directory_name_warnings: if False, will ensure the first directory name ends in 'noise' and the second
    directory name ends in 'blur'
    :return: effective entropy properties of the following form
        {shard_id_0: {
            entropy_function_tag_0_effective: [],
            entropy_function_tag_1_effective: [],
            ...
            }
        shard_id_1: {
            entropy_function_tag_0_effective: [],
            entropy_function_tag_1_effective: [],
            ...
            }
        ...
        }
    """

    relevant_directories = measured_dataset_directories[:2]
    # measured_dataset_directories starts at the end of the
    # distortion chain and works backwards, so the first two entries are always noise then blur

    if not ignore_directory_name_warnings:
        if str(relevant_directories[0])[-5:] != 'noise':
            raise Exception("Expected directory name ending in 'noise'")
        if str(relevant_directories[1])[-4:] != 'blur':
            raise Exception("Expected directory name ending in 'blur'")

    noise_directory = relevant_directories[0]
    blur_directory = relevant_directories[1]

    with open(Path(noise_directory, STANDARD_ENTROPY_PROPERTIES_FILENAME), 'r') as file:
        noise_stage_entropy_data = json.load(file)
        noise_stage_entropy_props = noise_stage_entropy_data['shard_entropy_properties']

    with open(Path(blur_directory, STANDARD_ENTROPY_PROPERTIES_FILENAME), 'r') as file:
        blur_stage_entropy_data = json.load(file)
        blur_stage_entropy_props = blur_stage_entropy_data['shard_entropy_properties']

    effective_entropy_properties = {}

    for shard_id in noise_stage_entropy_props.keys():

        noise_shard_properties = noise_stage_entropy_props[shard_id]
        blur_shard_properties = blur_stage_entropy_props[shard_id]

        for entropy_function_tag, noise_entropy in noise_shard_properties.items():
            noise_entropy = np.asarray(noise_entropy)
            blur_entropy = np.asarray(blur_shard_properties[entropy_function_tag])
            delta_entropy = noise_entropy - blur_entropy
            effective_entropy = blur_entropy - delta_entropy

            effective_entropy_function_tag = get_effective_entropy_equiv(entropy_function_tag)

            if shard_id not in effective_entropy_properties.keys():
                effective_entropy_properties[shard_id] = {
                    effective_entropy_function_tag: list(effective_entropy)
                }
            else:
                effective_entropy_properties[shard_id][effective_entropy_function_tag] = list(effective_entropy)

    return effective_entropy_properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='entropy_measurement_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'entropy_measurement_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    measure_dataset_entropy_properties(run_config)
