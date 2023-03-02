from pathlib import Path
from src.utils.definitions import ROOT_DIR, ORIGINAL_DATASETS
import json


def load_train2017_instances():

    train2017_rel_path = ORIGINAL_DATASETS['train2017']['rel_path']
    instances_filename = ORIGINAL_DATASETS['train2017']['metadata_filename']

    with open(Path(ROOT_DIR, train2017_rel_path, instances_filename), 'r') as f:
        instances = json.load(f)

    return instances


def get_coco128_image_ids():

    yolo_cfg = ORIGINAL_DATASETS['coco128']['yolo_cfg']
    dataset_rel_path = yolo_cfg['rel_path']
    image_dir = yolo_cfg['train']
    image_dir = Path(ROOT_DIR, dataset_rel_path, image_dir)

    image_paths = list(image_dir.iterdir())
    image_paths = [image_path for image_path in image_paths if image_path.name[-3:] in {'jpg', 'png'}]
    image_id_strings = [image_path.stem for image_path in image_paths]
    image_ids = set([int(image_id_string) for image_id_string in image_id_strings])

    return image_ids


def filter_images(images, image_ids):
    return [image for image in images if image['id'] in image_ids]


def filter_annotations(annotations, image_ids):
    return [annotation for annotation in annotations if annotation['image_id'] in image_ids]


def update_instances(instances, filtered_images, filtered_annotations):

    instances['images'] = filtered_images
    instances['annotations'] = filtered_annotations

    return instances


def log_coco128_instances(updated_instances):

    coco128_dir = ORIGINAL_DATASETS['coco128']['rel_path']
    target_directory = Path(ROOT_DIR, coco128_dir)
    filename = ORIGINAL_DATASETS['coco128']['metadata_filename']

    with open(Path(target_directory, filename), 'w') as f:
        json.dump(updated_instances, f)


def main():

    instances = load_train2017_instances()
    image_ids = get_coco128_image_ids()

    images = instances['images']
    images = filter_images(images, image_ids)
    annotations = instances['annotations']
    annotations = filter_annotations(annotations, image_ids)

    instances = update_instances(instances=instances,
                                 filtered_images=images,
                                 filtered_annotations=annotations)

    log_coco128_instances(instances)

if __name__ == '__main__':

    main()
