#!/usr/bin/env python
import json
from src.utils.functions import load_wandb_data_artifact, id_from_tags, get_config, \
    log_config
from src.utils.definitions import REL_PATHS, DATATYPE_MAP, STANDARD_DATASET_FILENAME, WANDB_PID, ROOT_DIR, \
    NATIVE_RESOLUTION
from src.pre_processing.build_sat6_np_dataset import mat_to_numpy
from src.pre_processing.build_places_np_dataset import transfer_to_numpy
from src.pre_processing.distortions import tag_to_image_distortion
from src.utils.functions import load_data_vectors
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import scipy.io
import wandb

wandb.login()


def distort_np_img_vec(vector_in, distortion_function, datatype_key, distortion_type_flag,
                       native_resolution=NATIVE_RESOLUTION):

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


def distort_dataset(dataset, new_dataset_path, distortion_tag, distortion_type_flag, datatype_key, dataset_split_key,
                    native_resolution=NATIVE_RESOLUTION):

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
                                                                       distortion_type_flag,
                                                                       native_resolution)
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

    if 'save_rgb_vector' in config.keys():
        save_rgb_vector = config['save_rgb_vector']  # used to save demo vector only
    else:
        save_rgb_vector = False

    if 'rgb' in config.keys():  # rgb throughout rather than pan
        rgb = config['rgb']
    else:
        rgb = False

    if rgb:
        pan_rgb_flag = 'rgb'
        convert_to_pan = False
    else:
        pan_rgb_flag = 'pan'
        convert_to_pan = True

    if rgb and save_rgb_vector:
        raise Exception('save_rgb_vector only intended for making visuals, not to be used with rgb datasets')

    with wandb.init(project=WANDB_PID, job_type='distort_dataset', tags=distortion_tags, notes=description,
                    config=config) as run:

        parent_artifact_name = f'{parent_dataset_id}:{parent_artifact_alias}'
        parent_artifact, parent_dataset = load_wandb_data_artifact(run,
                                                                   parent_artifact_name,
                                                                   parent_artifact_filename)
        if 'name_string' in config.keys() and config['name_string'] is not None:
            name_string = config['name_string']
            new_dataset_id, new_dataset_rel_dir = id_from_tags(artifact_type, [name_string], return_dir=True)
        else:
            new_dataset_id, new_dataset_rel_dir = id_from_tags(artifact_type, distortion_tags, return_dir=True)

        new_dataset_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir)
        Path.mkdir(new_dataset_abs_dir)

        run_metadata = config
        run_metadata_additions = {
            'artifact_filename': artifact_filename,
            'root_dir_at_run': str(ROOT_DIR),
            'dataset_rel_dir': str(new_dataset_rel_dir)
        }
        run_metadata.update(run_metadata_additions)

        if WANDB_PID[:4] == 'sat6':
            sat6 = True
            places = False
            data_abs_path = Path(ROOT_DIR, parent_dataset['dataset_rel_dir'], parent_dataset['dataset_filename'])
            data = scipy.io.loadmat(data_abs_path)
            data_x, data_y = data['test_x'], data['test_y']

        elif WANDB_PID[:6] == 'places':
            sat6 = False
            places = True
            image_shape = config['image_shape']
            parent_names_labels = parent_dataset['names_labels']
            starting_img_parent_rel_dir = parent_dataset['dataset_rel_dir']
        else:
            raise Exception('Invalid project ID')

        np0_image_label_filenames = []
        image_distortion_info = {}

        for i in range(iterations):
            file_count_offset = len(np0_image_label_filenames)

            if sat6:

                if save_rgb_vector:  # used only to save demo files locally

                    mat_to_numpy(data_x,
                                 data_y,
                                 num_images,
                                 images_per_file,
                                 datatype_key,
                                 # distortion_type_flags[0],
                                 'rgb',
                                 new_dataset_abs_dir,
                                 # res_distortion_func,
                                 file_count_offset=file_count_offset,
                                 filename_stem='test',
                                 parent_dataset_id=parent_artifact_name)

                if rgb:
                    raise Exception('rgb not implemented for SAT-6')

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
            elif places:

                if save_rgb_vector:  # used only to save demo files locally

                    transfer_to_numpy(parent_names_labels,
                                      starting_img_parent_rel_dir,
                                      num_images,
                                      images_per_file,
                                      image_shape,
                                      datatype_key,
                                      'rgb',
                                      new_dataset_abs_dir,
                                      convert_to_pan=False,
                                      file_count_offset=file_count_offset,
                                      filename_stem='test',
                                      parent_dataset_id=parent_dataset_id)

                new_data_subset = transfer_to_numpy(parent_names_labels,
                                                    starting_img_parent_rel_dir,
                                                    num_images,
                                                    images_per_file,
                                                    image_shape,
                                                    datatype_key,
                                                    pan_rgb_flag,
                                                    new_dataset_abs_dir,
                                                    convert_to_pan=convert_to_pan,
                                                    file_count_offset=file_count_offset,
                                                    filename_stem='test',
                                                    parent_dataset_id=parent_dataset_id)

            subset_image_label_filenames = new_data_subset['image_and_label_filenames']
            np0_image_label_filenames.extend(subset_image_label_filenames)
            parent_dataset_ids = new_data_subset['parent_dataset_ids']
            # image_distortion_info.update(res_data_subset['image_distortion_info'])

        np0_dataset = {dataset_split_key: {
            'image_and_label_filenames': np0_image_label_filenames,
            'image_distortion_info': image_distortion_info},
            'last_distortion_type_flag': pan_rgb_flag,  # enables subsequent dataset users to find correct subdirectory
            'parent_dataset_ids': parent_dataset_ids
        }

        np0_dataset.update(run_metadata)
        np0_artifact_id = f"{new_dataset_id}_{pan_rgb_flag}"
        np0_artifact = wandb.Artifact(np0_artifact_id,
                                      type=artifact_type,
                                      metadata=run_metadata)
        np0_artifact_path = Path(new_dataset_abs_dir, REL_PATHS[pan_rgb_flag], artifact_filename)
        with open(np0_artifact_path, 'w') as file:
            json.dump(np0_dataset, file)

        np0_artifact.add_file(str(np0_artifact_path))
        np0_artifact.metadata = run_metadata
        run.log_artifact(np0_artifact)
        np0_artifact.wait()  # allow artifact upload to finish before its used downstream

        # with the new pan artifact created, loop through distortion tags and create a new artifact for
        # the output of each associated distortion function, where each subsequent artifact has the last artifact
        # for its parent
        parent_dataset_id = np0_artifact_id
        parent_artifact_name = f'{np0_artifact_id}:latest'

        for i, distortion_tag in enumerate(distortion_tags):
            distortion_type_flag = distortion_type_flags[i]
            parent_artifact, parent_dataset = load_wandb_data_artifact(run,
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    distort_log_numpy(run_config)
