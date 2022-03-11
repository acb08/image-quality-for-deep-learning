#!/usr/bin/env python 
from __future__ import division
import wandb
import json
from PIL import Image
from torchvision import transforms
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, PROJECT_ID, STANDARD_DATASET_FILENAME, REL_PATHS
from src.d00_utils.functions import id_from_tags, get_config, read_json_artifact, string_from_tags
from src.d01_pre_processing.distortions import tag_to_transform

"""
Makes a distorted version of a dataset and logs as a W&B artifact 
"""

wandb.login()


def process_image(image_path, distortion_functions, tag_string, iteration_num=0):

    if CHECKOUT_MODE:
        return emulate_process_image(image_path, distortion_functions, tag_string, iteration_num=0)

    distortion_data = []
    image_name = image_path.stem
    extension = '.png'
    img = Image.open(image_path).convert('RGB')
    img = transforms.ToTensor()(img)

    for distortion_function in distortion_functions:
        img, distortion_tag, distortion_value = distortion_function(img)
        distortion_data.append((distortion_tag, distortion_value))

    image_name = image_name + tag_string + '_' + str(iteration_num) + extension
    img = transforms.ToPILImage()(img.squeeze_(0))

    return img, image_name, distortion_data


def emulate_process_image(image_path, distortion_functions, tag_string, iteration_num=0):

    distortion_data = []
    image_name = image_path.stem
    extension = '.checkout'

    for i in range(len(distortion_functions)):
        distortion_tag, distortion_value = f'pretend_tag_{i}', i
        distortion_data.append((distortion_tag, distortion_value))

    image_name = image_name + tag_string + str(iteration_num) + extension

    return 'placeholder', image_name, distortion_data


def distort_and_log(config):

    """
    Applies distortions to image dataset as specified in config, saves saves distortion data in a file specified
    by STANDARD_DATASET_FILENAME (dataset.json at time of doc string), uploads artifact to W&B with
    an artifact name where the first four characters are a sequential numerical identifier/key and the remainder
    of the name is concatenated distortion tags
    :param config: dict with distortion parameters
    :return:
    """

    parent_dataset_id = config['parent_dataset_id']
    description = config['description']
    artifact_type = config['artifact_type']
    artifact_filename = config['artifact_filename']
    distortion_tag_sets = config['distortion_tags']
    distortion_iterations = config['distortion_iterations']
    checkout_mode = config['checkout_mode']

    if len(distortion_tag_sets) != len(distortion_iterations):
        raise Exception('distortion tag and distortion iteration mismatch')

    parent_artifact_name = f'{parent_dataset_id}:latest'

    for distortion_key in distortion_tag_sets:

        distortion_tags = distortion_tag_sets[distortion_key]
        num_distortion_iterations = distortion_iterations[distortion_key]

        with wandb.init(project=PROJECT_ID, job_type='distort_dataset') as run:

            parent_artifact = run.use_artifact(parent_artifact_name)  # fixing name re-assigment
            parent_artifact_dir = parent_artifact.download()
            parent_dataset = read_json_artifact(parent_artifact_dir, artifact_filename)

            # consolidate parent and new distortion information for metadata
            parent_distortion_tags = parent_dataset['distortion_tags']
            all_distortion_tags = list(parent_distortion_tags)
            all_distortion_tags.extend(list(distortion_tags))
            parent_distortion_iterations = parent_dataset['distortion_iterations']
            all_distortion_iterations = list(parent_distortion_iterations)
            all_distortion_iterations.append(num_distortion_iterations)

            new_dataset_id = id_from_tags(artifact_type, all_distortion_tags)
            new_dataset_rel_parent_dir = REL_PATHS[artifact_type]
            new_dataset_rel_dir = Path(new_dataset_rel_parent_dir, new_dataset_id)

            new_dataset_abs_path = Path(ROOT_DIR, new_dataset_rel_dir)
            Path.mkdir(new_dataset_abs_path)
            full_dataset_path = Path(new_dataset_abs_path, artifact_filename)

            new_dataset_contents = distort(parent_dataset,
                                           distortion_tags,
                                           num_distortion_iterations,
                                           new_dataset_abs_path)

            run_metadata = config
            run_metadata_additions = {
                'artifact_filename': artifact_filename,
                'root_dir_at_run': str(ROOT_DIR),
                'checkout_mode': checkout_mode,
                'dataset_rel_dir': str(new_dataset_rel_dir),
            }
            run_metadata.update(run_metadata_additions)

            # log_metadata(artifact_type, new_dataset_id, run_metadata)  # remove redundant project metadata
            new_dataset_contents.update(run_metadata)

            artifact = wandb.Artifact(new_dataset_id,
                                      type=artifact_type,
                                      description=description,
                                      metadata=run_metadata
                                      )

            with open(full_dataset_path, 'w') as file:
                json.dump(new_dataset_contents, file)

            artifact.add_file(full_dataset_path)
            run.log_artifact(artifact)
            run.name = new_dataset_id
            wandb.finish()

        # reset parent_dataset_id and parent_artifact_name for next iteration of loop
        parent_dataset_id = new_dataset_id
        parent_artifact_name = f'{parent_dataset_id}:latest'


def distort(starting_dataset, distortion_tags, num_distortion_iterations, new_dataset_path):

    """
    transforms the dataset captured in starting_dataset by applying the distortion functions associated
    with distortion_tags and saving images to a new directory

    :param starting_dataset: dictionary containing all dataset info/metadata except the images themselves
    :param distortion_tags: new distortions to be applied to the images
    :param num_distortion_iterations: number of times each set of distortion functions is applied
    :param new_dataset_path: path to directory for new images

    :return: new_dataset (dictionary) containing list of (name, label) tuples and new_image_distortion_info dictionary
    containing distortion data keyed by image names
    """

    tag_string = string_from_tags(distortion_tags)
    starting_img_parent_rel_dir = starting_dataset['dataset_rel_dir']
    starting_img_rel_dir = Path(starting_img_parent_rel_dir, REL_PATHS['images'])
    starting_img_dir = Path(ROOT_DIR, starting_img_rel_dir)
    parent_names_labels = starting_dataset['names_labels']
    image_distortion_info_key = 'image_distortion_info'

    new_image_dir = Path(new_dataset_path, REL_PATHS['images'])
    if not new_image_dir.is_dir():
        Path.mkdir(new_image_dir)

    distortion_functions = []
    for tag in distortion_tags:
        distortion_functions.append(tag_to_transform(tag))

    new_names_labels = []
    new_image_distortion_info = {}

    if image_distortion_info_key not in starting_dataset.keys():
        inherited_distortion = False
    else:
        image_distortion_info = starting_dataset[image_distortion_info_key].copy()
        inherited_distortion = True

    for iteration in range(num_distortion_iterations):

        for (img_name, img_label) in parent_names_labels:

            img_path = Path(starting_img_dir, img_name)
            new_img, new_name, distortion_data = process_image(img_path,
                                                               distortion_functions,
                                                               tag_string,
                                                               iteration_num=iteration)
            if not CHECKOUT_MODE:
                new_img.save(Path(new_image_dir, new_name))

            new_names_labels.append((new_name, img_label))

            if inherited_distortion:
                parent_img_distortion_dict = image_distortion_info[img_name].copy()
                img_distortion_params = parent_img_distortion_dict['distortion_params']
                parent_images = parent_img_distortion_dict['parent_images']
            else:
                img_distortion_params = []
                parent_images = []

            img_distortion_params.extend(distortion_data)
            parent_images.append((img_name, starting_img_parent_rel_dir))

            new_image_distortion_info[new_name] = {
                'parent_images': parent_images,
                'distortion_params': img_distortion_params
            }

    new_dataset = {
        'names_labels': new_names_labels,
        'image_distortion_info': new_image_distortion_info,
    }

    return new_dataset


if __name__ == '__main__':

    _cwd = Path.cwd()
    _manual_config = True
    CHECKOUT_MODE = True

    if _manual_config:

        _description = ''  # text for misc notes on artifact
        _config = {
            'parent_dataset_id': 'val_256',  # dataset to be distorted, key in json metadata file *and* W&D artifact ID
            'artifact_type': 'test_dataset',
            'description': _description,
            'artifact_filename': STANDARD_DATASET_FILENAME,  # do not recommend changing
            'checkout_mode': CHECKOUT_MODE,
            'distortion_tags': {
                'pre_noise': ('pan', ),
                'noise': ('haha',)
            },
            'distortion_iterations': {
                'pre_noise': 1,
                'noise': 1
            },
        }

        print('running with manual config')

    else:
        _config = get_config(_cwd)  # TODO: update with argparse to specify in bash script which config filename to use

    distort_and_log(_config)
