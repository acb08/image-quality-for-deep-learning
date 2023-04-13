from PIL import ImageDraw
from src.utils.definitions import STANDARD_DATASET_FILENAME
import json
from pathlib import Path
from src.utils.definitions import ROOT_DIR
from src.utils.classes import Illustrator
import yaml


def load_dataset(directory):
    with open(Path(directory, STANDARD_DATASET_FILENAME), 'r') as file:
        dataset = json.load(file)
    return dataset


def get_illustration_dataset(dataset):
    instances = dataset['instances']
    image_dir = Path(ROOT_DIR, dataset['dataset_rel_dir'])
    illustrator = Illustrator(image_dir, instances)
    return illustrator


def illustrate_image(image, boxes):

    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline='red', width=2)


def illustrate(directory, output_directory='illustrated', log_licenses=True):

    directory = Path(directory).expanduser()
    output_directory = Path(output_directory)
    if not output_directory.is_absolute():
        output_directory = Path(directory, output_directory)
        if not output_directory. is_dir():
            output_directory. mkdir()

    dataset = load_dataset(directory)
    illustration_dataset = get_illustration_dataset(dataset)

    for image, boxes, filename in illustration_dataset:
        illustrate_image(image, boxes['boxes'])
        image.save(Path(output_directory, filename))

    if log_licenses:
        license_url_map = illustration_dataset.license_url_map()
        with open(Path(output_directory, 'license_map.yml'), 'w') as f:
            yaml.dump(license_url_map, f)


if __name__ == '__main__':

    _directory = r'/home/acb6595/coco/datasets/demo/0001demo-coco_mp90_demo/images/test'
    illustrate(_directory)
