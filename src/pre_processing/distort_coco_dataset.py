#!/usr/bin/env python
import copy
import json
from src.utils.functions import load_wandb_data_artifact, id_from_tags, get_config, \
    log_config, string_from_tags
from src.utils.definitions import REL_PATHS, STANDARD_DATASET_FILENAME, WANDB_PID, ROOT_DIR
from src.pre_processing.distortions import coco_tag_to_image_distortions
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import wandb
from src.utils import detection_functions
import random
import time
import shutil
wandb.login()

FRAGILE_ANNOTATION_KEYS = ['area', 'segmentation']


def resize_bbox(res_frac, bbox):

    bbox = np.asarray(bbox)
    bbox = res_frac * bbox
    bbox = list(bbox)

    return bbox


def make_train_val_splits(instances, val_frac):

    images = instances['images']
    num_val = int(val_frac * len(images))

    random.shuffle(images)
    val_images = images[:num_val]
    train_images = images[num_val:]
    assert len(train_images) + len(val_images) == len(images)

    train_image_ids = [image['id'] for image in train_images]
    train_image_ids = set(train_image_ids)
    val_image_ids = [image['id'] for image in val_images]
    val_image_ids = set(val_image_ids)
    assert len(train_image_ids.intersection(val_image_ids)) == 0

    annotations = instances['annotations']
    train_annotations = [annotation for annotation in annotations if annotation['image_id'] in train_image_ids]
    val_annotations = [annotation for annotation in annotations if annotation['image_id'] in val_image_ids]
    assert len(train_annotations) + len(val_annotations) == len(annotations)

    val_annotation_ids = [annotation['id'] for annotation in val_annotations]
    val_annotation_ids = set(val_annotation_ids)
    train_annotation_ids = [annotation['id'] for annotation in train_annotations]
    train_annotation_ids = set(train_annotation_ids)
    assert len(train_annotation_ids.intersection(val_annotation_ids)) == 0

    instances.pop('images')
    instances.pop('annotations')

    train_instances = copy.deepcopy(instances)
    val_instances = copy.deepcopy(instances)

    train_instances['images'] = train_images
    train_instances['annotations'] = train_annotations

    val_instances['images'] = val_images
    val_instances['annotations'] = val_annotations

    return train_instances, val_instances


def apply_distortions(image, distortion_functions, mapped_annotations, updated_image_id,
                      remove_fragile_annotations=True):
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

        image, __, distortion_type_flag, distortion_value = distortion_func(image)

        if distortion_type_flag == 'res':
            res = distortion_value
            updated_mapped_annotations = update_annotations(mapped_annotations,
                                                            res=res,
                                                            updated_image_id=updated_image_id,
                                                            remove_fragile_annotations=remove_fragile_annotations)

        distortion_data[distortion_type_flag] = distortion_value

    if updated_mapped_annotations is None:  # inserts updated image id, leaves bounding boxes unchanged when res == 1
        updated_mapped_annotations = update_annotations(mapped_annotations,
                                                        res=1,
                                                        updated_image_id=updated_image_id,
                                                        remove_fragile_annotations=remove_fragile_annotations)

    return image, updated_mapped_annotations, distortion_data


def update_annotations(annotations, res, updated_image_id, remove_fragile_annotations=True):

    """
    Updates the annotations associated with an image that has been distorted.  Updates the image_id and wWhen
    resolution != 1, updates the bounding boxes.
    """

    updated_annotations = []

    for annotation in annotations:

        updated_annotation = copy.deepcopy(annotation)

        if res != 1:
            bbox = annotation['bbox']
            bbox = resize_bbox(res, bbox)
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


def image_to_yolo_label_filename(image_filename):
    return Path(image_filename).stem + '.txt'


def distort_coco(image_directory, instances, iterations, distortion_tags, output_dir, cutoff=None,
                 yolo_parent_label_dir=None,
                 dataset_split_key=None):

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
    :param yolo_parent_label_dir: path to directory of yolo label files
    :param dataset_split_key: str, 'train', 'val', or 'test'
    :return:
    """

    parent_images = instances['images']
    if yolo_parent_label_dir is not None:
        yolo_label_output_dir = image_dir_to_yolo_label_dir(output_dir)
    else:
        yolo_label_output_dir = None

    if cutoff == 'all':
        cutoff = None

    if cutoff is not None:
        parent_images = parent_images[:cutoff]
    parent_annotations = instances['annotations']
    print('got annotations')
    parent_image_ids = detection_functions.get_image_ids(parent_images)
    print('got parent image ids')
    mapped_parent_annotations = detection_functions.map_annotations(parent_annotations, parent_image_ids,
                                                                    give_status=True)

    distortion_functions = []
    for tag in distortion_tags:
        distortion_function = coco_tag_to_image_distortions[tag]
        distortion_functions.append(distortion_function)

    new_image_ids = set()
    images = []
    annotations = []

    for i in range(iterations):

        print(f'iteration {i + 1} in progress, {round(time.time() - T_0, 2)} seconds')

        for j, parent_img_data in enumerate(parent_images):

            if j < 100 and j % 5 == 0:
                print(f'image {j + 1}, {round(time.time() - T_0, 2)} seconds')

            parent_file_name = parent_img_data['file_name']
            parent_image_id = parent_img_data['id']
            parent_annotations = mapped_parent_annotations[parent_image_id]
            parent_image = Image.open(Path(image_directory, parent_file_name))

            image_data = copy.deepcopy(parent_img_data)
            new_id = new_image_id(new_image_ids)
            new_image_ids.add(new_id)
            name_stem = str(new_id).rjust(12, '0')
            file_name = name_stem + '.png'

            if yolo_label_output_dir is not None:
                parent_label_file_name = str(Path(parent_file_name).stem) + '.txt'
                parent_label_file_path = Path(yolo_parent_label_dir, parent_label_file_name)
                new_label_file_name = name_stem + '.txt'
                new_label_file_path = Path(yolo_label_output_dir, new_label_file_name)
                try:
                    shutil.copy(parent_label_file_path, new_label_file_path)
                except FileNotFoundError:
                    pass

            image, updated_annotations, distortion_data = apply_distortions(image=parent_image,
                                                                            distortion_functions=distortion_functions,
                                                                            mapped_annotations=parent_annotations,
                                                                            updated_image_id=new_id)

            if type(image) != Image.Image:
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

            if (j + 1) % 500 == 0:
                print(f'{j+1} images processed')

    new_instances = copy.deepcopy(instances)
    new_instances['images'] = images
    new_instances['annotations'] = annotations

    return new_instances


# def get_dataset_split_keys(artifact_type):
#     if artifact_type == 'train_dataset':
#         return 'train', 'val'
#     elif artifact_type == 'test_dataset':
#         return 'test'
#     else:
#         return None


# def directory_sub_structure(artifact_type):
#
#     if artifact_type == 'train_dataset':
#         dataset_split_keys = ('train', 'val')
#     elif artifact_type == 'test_dataset':
#         dataset_split_keys = ('test', )
#     else:
#         raise ValueError('directory_sub_structure only defined for train and test datasets')
#
#     dataset_sub_struct = REL_PATHS['dataset_sub_struct']
#
#     return dataset_split_keys, {key: dataset_sub_struct[key] for key in dataset_split_keys}\


# def find_yolo_labels(image_dir, image_suffixes=('.png', '.jpg'), intermediate_sub_dir='labels'):
#
#     image_suffixes = set(image_suffixes)
#     image_names = get_relevant_filenames(image_dir, extensions=image_suffixes)
#     image_names = set(image_names)
#
#     yolo_label_filenames = [image_to_yolo_label_filename(name) for name in image_names]
#     yolo_label_filenames = set(yolo_label_filenames)
#
#     target_sub_dir_name = Path(image_dir).parts[-1]
#
#     found = False
#     search_dir = Path(image_dir)
#     searched_to_home_dir = False
#
#     while not found and not searched_to_home_dir:
#
#         if Path(search_dir, target_sub_dir_name).is_dir():
#             pass
#
#
# def check_matching_filenames(directory, target_filenames):
#     directory = Path(directory)
#
#
# def get_relevant_filenames(directory, extensions):
#
#     extensions = set(extensions)
#     file_paths = list(Path(directory).iterdir())
#     file_names = [file_path.name for file_path in file_paths if file_path.suffix in extensions]
#
#     return file_names
#


def image_dir_to_yolo_label_dir(image_path):

    image_path = Path(image_path)
    path_parts = image_path.parts
    image_indices = [i for i, part in enumerate(path_parts) if part == 'images']

    try:
        last_image_idx = max(image_indices)
        path_parts = list(path_parts)
        path_parts[last_image_idx] = 'labels'
        return Path(*path_parts)

    except ValueError:
        return None


def image_dir_yolo_text_file(text_file_path, max_lines=100):

    with open(text_file_path, 'r') as f:

        for i, line in enumerate(f):
            if line[-4:] in {'.jpg', '.png'}:
                break
            elif i >= max_lines:
                raise ValueError(f'no image paths found, last line: {line}')

    return str(Path(line).parent)


def distort_log_coco(config):

    parent_dataset_id = config['parent_dataset_id']
    parent_artifact_alias = config['parent_artifact_alias']
    parent_artifact_filename = config['parent_artifact_filename']
    if parent_artifact_filename == 'standard':
        parent_artifact_filename = STANDARD_DATASET_FILENAME
    artifact_type = config['artifact_type']
    num_images = config['num_images']

    artifact_filename = config['artifact_filename']
    if artifact_filename == 'standard':
        artifact_filename = STANDARD_DATASET_FILENAME
    distortion_tags = config['distortion_tags']
    distortion_type_flags = config['distortion_type_flags']
    iterations = config['iterations']
    description = config['description']

    if artifact_type == 'train_dataset':
        val_frac = config['val_frac']
    else:
        val_frac = None

    # dataset_split_keys, dataset_sub_dirs = directory_sub_structure(artifact_type)

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

        _new_dataset_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir)
        Path.mkdir(_new_dataset_abs_dir)

        _new_dataset_common = {

            'parent_dataset_id': parent_dataset_id,
            'description': description,
            'artifact_filename': artifact_filename,
            'distortion_tags': distortion_tags,
            'distortion_type_flags': distortion_type_flags,
            'distortion_iterations': iterations,
            'ROOT_DIR_at_run': ROOT_DIR,

        }

        # get image/label sub-directories here

        run_metadata = config
        run_metadata_additions = {
            'artifact_filename': artifact_filename,
            'root_dir_at_run': str(ROOT_DIR),
            'dataset_rel_dir': str(new_dataset_rel_dir)
        }
        run_metadata.update(run_metadata_additions)

        parent_dataset_dir = parent_dataset['dataset_dir']
        parent_dataset_abs_dir = Path(ROOT_DIR, parent_dataset_dir)  # always the image directory with dataset.json
        yolo_label_dir = image_dir_to_yolo_label_dir(parent_dataset_abs_dir)

        instances = parent_dataset['instances']
        # if 'yolo_cfg' in parent_dataset.keys():
        #     parent_yolo_config = parent_dataset['yolo_cfg']
        # else:
        #     parent_yolo_cfg = None

        yolo_cfg = {'path': str(new_dataset_rel_dir)}

        if artifact_type == 'train_dataset':

            val_dataset_id = f'{new_dataset_id}-val_split'
            print('new val dataset id:', val_dataset_id, round(time.time() - T_0, 2), 'seconds')

            dataset_split_keys = ('train', 'val')  # should match keys in parent_yolo_cfg
            train_dataset_split_key, val_dataset_split_key = dataset_split_keys

            train_sub_dirs = REL_PATHS['dataset_sub_struct'][train_dataset_split_key]
            train_image_sub_dir, train_label_sub_dir = train_sub_dirs
            train_image_rel_dir = Path(new_dataset_rel_dir, train_image_sub_dir)

            train_image_abs_dir = Path(ROOT_DIR, train_image_rel_dir)
            train_label_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir, train_label_sub_dir)
            train_image_abs_dir.mkdir(parents=True)
            train_label_abs_dir.mkdir(parents=True)

            val_sub_dirs = REL_PATHS['dataset_sub_struct'][val_dataset_split_key]
            val_image_sub_dir, val_label_sub_dir = val_sub_dirs
            val_image_rel_dir = Path(new_dataset_rel_dir, val_image_sub_dir)

            val_image_abs_dir = Path(ROOT_DIR, val_image_rel_dir)
            val_label_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir, val_label_sub_dir)
            val_image_abs_dir.mkdir(parents=True)
            val_label_abs_dir.mkdir(parents=True)

            print('beginning train/val splitting', round(time.time() - T_0, 2), 'seconds')
            train_instances, val_instances = make_train_val_splits(instances, val_frac=val_frac)
            print('beginning train distortions', round(time.time() - T_0, 2), 'seconds')
            new_train_instances = distort_coco(image_directory=parent_dataset_abs_dir,
                                               instances=train_instances,
                                               iterations=iterations,
                                               distortion_tags=distortion_tags,
                                               output_dir=train_image_abs_dir,
                                               cutoff=num_images,
                                               yolo_parent_label_dir=yolo_label_dir,
                                               dataset_split_key=train_dataset_split_key)

            yolo_train_path_pointer = Path(train_image_abs_dir).relative_to(_new_dataset_abs_dir)
            yolo_cfg[train_dataset_split_key] = str(yolo_train_path_pointer)

            new_train_dataset = copy.deepcopy(_new_dataset_common)  # need to add dataset_rel_dir, instances, yolo_cfg
            new_train_dataset['dataset_rel_dir'] = str(train_image_rel_dir)
            new_train_dataset['instances'] = new_train_instances
            new_train_dataset['val_sibling_dataset_id'] = val_dataset_id
            print('beginning val distortions', round(time.time() - T_0, 2), 'seconds')
            new_val_instances = distort_coco(image_directory=parent_dataset_abs_dir,
                                             instances=val_instances,
                                             iterations=iterations,
                                             distortion_tags=distortion_tags,
                                             output_dir=val_image_abs_dir,
                                             cutoff=num_images,
                                             yolo_parent_label_dir=yolo_label_dir,
                                             dataset_split_key=val_dataset_split_key)

            yolo_val_path_pointer = Path(val_image_abs_dir).relative_to(_new_dataset_abs_dir)
            yolo_cfg[val_dataset_split_key] = str(yolo_val_path_pointer)

            new_val_dataset = copy.deepcopy(_new_dataset_common)
            new_val_dataset['instances'] = new_val_instances
            new_val_dataset['dataset_rel_dir'] = str(val_image_rel_dir)
            new_val_dataset['yolo_cfg'] = None  # relevant yolo_cfg in sibling train dataset

            new_train_dataset['yolo_cfg'] = yolo_cfg

            new_train_artifact = wandb.Artifact(new_dataset_id,
                                                type=artifact_type,
                                                metadata=run_metadata)
            new_train_artifact_path = Path(train_image_abs_dir, artifact_filename)
            with open(new_train_artifact_path, 'w') as file:
                json.dump(new_train_dataset, file)
            new_train_artifact.add_file(str(new_train_artifact_path))
            new_train_artifact.metadata = run_metadata
            run.log_artifact(new_train_artifact)
            new_train_artifact.wait()

            new_val_artifact = wandb.Artifact(val_dataset_id,
                                              type='val_dataset',
                                              metadata=run_metadata)
            new_val_artifact_path = Path(val_image_abs_dir, artifact_filename)
            with open(new_val_artifact_path, 'w') as file:
                json.dump(new_val_dataset, file)
            new_val_artifact.add_file(str(new_train_artifact_path))
            new_val_artifact.metadata = run_metadata
            run.log_artifact(new_val_artifact)
            new_val_artifact.wait()

        elif artifact_type == 'test_dataset':

            dataset_split_keys = ('test', )  # should match keys in parent_yolo_cfg
            test_dataset_split_key = dataset_split_keys[0]
            test_sub_dirs = REL_PATHS['dataset_sub_struct'][test_dataset_split_key]
            test_image_sub_dir, test_label_sub_dir = test_sub_dirs
            test_image_rel_dir = Path(new_dataset_rel_dir, test_image_sub_dir)

            test_image_abs_dir = Path(ROOT_DIR, test_image_rel_dir)
            test_label_abs_dir = Path(ROOT_DIR, new_dataset_rel_dir, test_label_sub_dir)
            test_image_abs_dir.mkdir(parents=True)
            test_label_abs_dir.mkdir(parents=True)

            new_test_instances = distort_coco(image_directory=parent_dataset_abs_dir,
                                              instances=instances,
                                              iterations=iterations,
                                              distortion_tags=distortion_tags,
                                              output_dir=test_image_abs_dir,
                                              cutoff=num_images,
                                              yolo_parent_label_dir=yolo_label_dir)
            yolo_test_path_pointer = Path(test_image_abs_dir).relative_to(_new_dataset_abs_dir)
            yolo_cfg[test_dataset_split_key] = str(yolo_test_path_pointer)

            new_test_dataset = copy.deepcopy(_new_dataset_common)  # need to add dataset_rel_dir, instances, yolo_cfg
            new_test_dataset['dataset_rel_dir'] = str(test_image_rel_dir)
            new_test_dataset['instances'] = new_test_instances
            new_test_dataset['yolo_cfg'] = yolo_cfg

            new_test_artifact = wandb.Artifact(new_dataset_id,
                                               type=artifact_type,
                                               metadata=run_metadata)
            new_test_artifact_path = Path(test_image_abs_dir, artifact_filename)
            with open(new_test_artifact_path, 'w') as file:
                json.dump(new_test_dataset, file)
            new_test_artifact.add_file(str(new_test_artifact_path))
            new_test_artifact.metadata = run_metadata
            run.log_artifact(new_test_artifact)
            new_test_artifact.wait()

        else:
            raise ValueError('artifact_type must be either train_dataset or test_dataset')

        run.name = new_dataset_id
        log_config(_new_dataset_abs_dir, dict(config), return_path=False)

        # new_dataset = {
        #     'dataset_rel_dir': str(new_dataset_rel_dir),
        #
        #     'parent_dataset_id': parent_dataset_id,
        #     'description': description,
        #     'artifact_filename': artifact_filename,
        #     'distortion_tags': distortion_tags,
        #     'distortion_type_flags': distortion_type_flags,
        #     'distortion_iterations': iterations,
        #     'ROOT_DIR_at_run': ROOT_DIR,
        #
        #     'yolo_cfg': yolo_cfg
        # }

        # name_tags = [new_dataset_id]
        # name_tags.extend(distortion_type_flags)
        # new_artifact_id = string_from_tags(name_tags)


if __name__ == '__main__':

    T_0 = time.time()

    _distortion_config_filename = 'coco_noise_train.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=_distortion_config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_configs_detection'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    distort_log_coco(run_config)

    _t_f = time.time()
    print(f'complete, total time = {_t_f - T_0}')

