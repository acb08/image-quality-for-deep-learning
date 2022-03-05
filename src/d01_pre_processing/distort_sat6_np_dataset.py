#!/usr/bin/env python
import json

from src.d00_utils.functions import load_wandb_artifact_model, load_wandb_dataset_artifact, id_from_tags, get_config, \
    log_config
from src.d00_utils.definitions import REL_PATHS, DATATYPE_MAP, STANDARD_DATASET_FILENAME, PROJECT_ID, ROOT_DIR
from src.d01_pre_processing.build_sat6_np_dataset import mat_to_numpy
from src.d01_pre_processing.distortions import tag_to_func
from src.d02_train.train import load_data_vectors
import numpy as np
import argparse
from pathlib import Path
import scipy.io
import wandb

wandb.login()


def distort_blur_noise(distortion_tag, use_tag, parent_artifact_id, parent_artifact_alias):




    pass


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
    use_flags = config['use_flags']
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

        # resolution distortion performed at this level of function since it starts with the original .mat dataset,
        # whereas the blur and noise distortions can be handled from a common function
        res_distortion_func = tag_to_func[distortion_tags[0]]()
        res_image_label_filenames = []
        image_distortion_info = {}

        for i in range(iterations):

            file_count_offset = len(res_image_label_filenames)
            res_data_subset = mat_to_numpy(data_x,
                                           data_y,
                                           num_images,
                                           images_per_file,
                                           datatype_key,
                                           use_flags[0],
                                           new_dataset_abs_dir,
                                           res_distortion_func,
                                           file_count_offset=file_count_offset,
                                           filename_stem='test',
                                           parent_dataset_id=parent_artifact_name)

            subset_image_label_filenames = res_data_subset['image_and_label_filenames']
            res_image_label_filenames.extend(subset_image_label_filenames)
            image_distortion_info.update(res_data_subset['image_distortion_info'])

        res_dataset = {use_flags[0]: {
            'image_and_label_filenames': res_image_label_filenames,
            'image_distortion_info': image_distortion_info}
        }

        res_dataset.update(run_metadata)
        res_artifact_id = f"{new_dataset_id}_{use_flags[0]}"
        res_artifact = wandb.Artifact(res_artifact_id,
                                      type=artifact_type,
                                      metadata=run_metadata)
        res_artifact_path = Path(new_dataset_abs_dir, REL_PATHS[use_flags[0]], artifact_filename)
        with open(res_artifact_path, 'w') as file:
            json.dump(res_dataset, file)

        res_artifact.add_file(res_artifact_path)
        res_artifact.metadata = run_metadata
        run.log_artifact(res_artifact)
        run.name = new_dataset_id

    return res_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    print(run_config)

    res_test_dataset = distort_log_numpy(run_config)
