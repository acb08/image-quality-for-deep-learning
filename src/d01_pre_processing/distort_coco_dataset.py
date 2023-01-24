#!/usr/bin/env python
import copy
import json
from src.d00_utils.functions import load_wandb_data_artifact, id_from_tags, get_config, \
    log_config, string_from_tags
from src.d00_utils.definitions import REL_PATHS, DATATYPE_MAP, STANDARD_DATASET_FILENAME, WANDB_PID, ROOT_DIR, \
    NATIVE_RESOLUTION
from src.d01_pre_processing.distortions import tag_to_image_distortion
from src.d00_utils.functions import load_data_vectors
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import scipy.io
import wandb
from src.d00_utils import detection_functions
import random
wandb.login()


FRAGILE_ANNOTATION_KEYS = ['area', 'segmentation']


def apply_distortions(image, distortion_functions, mapped_annotations, updated_image_id, remove_fragile_annotations=True):
    """

    :param image: image, PIL or numpy array
    :param distortion_functions: list, image distortion functions
    :param mapped_annotations: list containing each annotation dict assocated with the image to be distorted
      {
      'boxes': [[x, y, width, height], [x, y, width, height], ...]
      'labels': [label_0, label_1, ...]
      'id': int, (unique id for the annotation itself)
      'category_id': int, (label for the object in bounding box)
      'image_id': int, (links annotation to associated image, coco_annotation['image_id'] <---> coco_img_data['id'])},
    :param updated_image_id: int, updated id for distorted image, which must replace the image_id in the original
    mapped_annotations
    :param remove_fragile_annotations: bool, if True all annotation fields that may be affected by bounding box
    adjustment are removed
    :return:
        image: PIL, distorted version of input image
        mapped_annotation: dict, contains updated bounding boxes (list) and associated labels (list)
        distortion_data: dict, contains distortion type tags and associated values
    """

    distortion_data = {}
    updated_mapped_annotations = None

    for distortion_func in distortion_functions:
        image, bbox_adjustment_func, distortion_type, distortion_value = distortion_func(image)

        updated_mapped_annotations = update_annotations(mapped_annotations,
                                                        bbox_adjustment_func=bbox_adjustment_func,
                                                        updated_image_id=updated_image_id,
                                                        remove_fragile_annotations=remove_fragile_annotations)
        distortion_data[distortion_type] = distortion_value

    # if updated_mapped_annotations is None:
    #     updated_mapped_annotations = copy.deepcopy(mapped_annotations)

    return image, updated_mapped_annotations, distortion_data


def update_annotations(annotations, bbox_adjustment_func, updated_image_id, remove_fragile_annotations=True):

    updated_annotations = []

    for annotation in annotations:

        updated_annotation = copy.deepcopy(annotation)
        bbox = annotation['bbox']
        if bbox_adjustment_func is not None:
            bbox = bbox_adjustment_func(bbox)
        updated_annotation['bbox'] = bbox
        updated_annotation['image_id'] = updated_image_id

        if remove_fragile_annotations:
            for key in FRAGILE_ANNOTATION_KEYS:
                if key in updated_annotation.keys():
                    updated_annotation.pop(key)

        updated_annotations.append(updated_annotation)

    return updated_annotations


def new_image_id(existing_ids, low=0, high=int(1e12)):
    new_id = random.randint(low, high)
    if new_id not in existing_ids:
        return new_id
    else:
        print(f'fyi: image id collision ({new_id})')
        return new_image_id(existing_ids)


def distort_coco(image_directory, instances, iterations, distortion_tags, output_dir, cutoff=None):

    """
    :param image_directory: str or Path object
    :param instances: dict
        {'images': [ --> list of dicts
             {'height': h (pixels),
            'width': w (pixels),
            'id': image id used to map annotations (e.g. bounding boxes) to associated images,
            }, plus additional metadata such as the image license and flickr url
            {}, {}, ...]
        'annotations': [ --> list of dicts

            {}. {}, ...]
        other keys that are not important here...
    :param iterations: int, number of distorted copies of original input images to make
    :param distortion_tags: list of tags that can be mapped to distortion functions
    :param output_dir: str or Path object
    :param cutoff: number of images in original dataset to use

    :return:
    """

    parent_images = instances['images']
    if cutoff is not None:
        parent_images = parent_images[:cutoff]
    parent_annotations = instances['annotations']
    parent_image_ids = detection_functions.get_image_ids(parent_images)
    mapped_parent_annotations = detection_functions.map_annotations(parent_annotations, parent_image_ids)

    distortion_functions = []
    for tag in distortion_tags:
        distortion_function = tag_to_image_distortion[tag]
        distortion_functions.append(distortion_function)

    new_image_ids = set()
    images = []
    annotations = []

    for i in range(iterations):

        for j, parent_img_data in enumerate(parent_images):

            parent_file_name = parent_img_data['file_name']
            parent_image_id = parent_img_data['id']
            parent_annotations = mapped_parent_annotations[parent_image_id]
            parent_image = Image.open(Path(image_directory, parent_file_name))

            image_data = copy.deepcopy(parent_img_data)
            new_id = new_image_id(new_image_ids)
            new_image_ids.add(new_id)
            file_name = str(new_id).rjust(12, '0') + '.png'

            image, updated_annotations, distortion_data = apply_distortions(image=parent_image,
                                                                            distortion_functions=distortion_functions,
                                                                            mapped_annotations=parent_annotations,
                                                                            updated_image_id=new_id)

            try:
                image.save(Path(output_dir, file_name))
            except AttributeError:
                image = Image.fromarray(image)
                image.save(Path(output_dir, file_name))

            height, width = np.shape(image)[:2]

            image_data.update(distortion_data)
            image_data['id'] = new_id
            image_data['parent_id'] = parent_image_id
            image_data['width'] = width
            image_data['height'] = height
            image_data['file_name'] = file_name

            images.append(image_data)
            annotations.extend(updated_annotations)

    new_instances = copy.deepcopy(instances)
    new_instances['images'] = images
    new_instances['annotations'] = annotations

    return new_instances


def distort_log_coco(config):

    parent_dataset_id = config['parent_dataset_id']
    parent_artifact_alias = config['parent_artifact_alias']
    parent_artifact_filename = config['parent_artifact_filename']
    if parent_artifact_filename == 'standard':
        parent_artifact_filename = STANDARD_DATASET_FILENAME
    artifact_type = config['artifact_type']
    num_images = config['num_images']

    # datatype_key = config['datatype_key']
    artifact_filename = config['artifact_filename']
    if artifact_filename == 'standard':
        artifact_filename = STANDARD_DATASET_FILENAME
    distortion_tags = config['distortion_tags']
    distortion_type_flags = config['distortion_type_flags']
    # dataset_split_key = config['dataset_split_key']
    iterations = config['iterations']
    description = config['description']

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

        parent_dataset_dir = parent_dataset['dataset_dir']
        parent_dataset_abs_dir = Path(ROOT_DIR, parent_dataset_dir)

        instances = parent_dataset['instances']

        new_instances = distort_coco(image_directory=parent_dataset_abs_dir,
                                     instances=instances,
                                     iterations=iterations,
                                     distortion_tags=distortion_tags,
                                     output_dir=new_dataset_abs_dir,
                                     cutoff=num_images)

        new_dataset = {
            'dataset_rel_dir': str(new_dataset_rel_dir),
            'instances': new_instances,
            'parent_dataset_id': parent_dataset_id,
            'description': description,
            'artifact_filename': artifact_filename,
            'distortion_tags': distortion_tags,
            'distortion_iterations': iterations,
            'ROOT_DIR_at_run': ROOT_DIR
        }

        name_tags = [new_dataset_id]
        name_tags.extend(distortion_type_flags)
        new_artifact_id = string_from_tags(name_tags)
        new_artifact = wandb.Artifact(new_artifact_id,
                                      type=artifact_type,
                                      metadata=run_metadata)
        new_artifact_path = Path(new_dataset_abs_dir, artifact_filename)
        with open(new_artifact_path, 'w') as file:
            json.dump(new_dataset, file)

        new_artifact.add_file(new_artifact_path)
        new_artifact.metadata = run_metadata
        run.log_artifact(new_artifact)
        new_artifact.wait()

        run.name = new_dataset_id
        log_config(new_dataset_abs_dir, dict(config), return_path=False)


if __name__ == '__main__':

    _distortion_config_filename = 'coco_val2017_mini.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=_distortion_config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    distort_log_coco(run_config)
