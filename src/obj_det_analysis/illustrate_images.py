import copy

from PIL import ImageDraw
from src.utils.definitions import STANDARD_DATASET_FILENAME
import json
from pathlib import Path
from src.utils.definitions import ROOT_DIR
from src.utils.classes import Illustrator
import yaml


def load_dataset(directory, dataset_in_parent_dir=False):

    if dataset_in_parent_dir:
        directory = Path(directory).parent

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


def abridged_dataset_transfer(start_directory, target_directory, pop_keys=('res', 'blur', 'noise')):

    dataset = load_dataset(start_directory)
    abridged_dataset = copy.deepcopy(dataset)

    if Path(target_directory).is_absolute():
        target_directory = Path(target_directory).relative_to(ROOT_DIR)
    abridged_dataset['dataset_rel_dir'] = str(target_directory)

    for image in abridged_dataset['instances']['images']:
        for key in pop_keys:
            try:
                image.pop(key)
            except KeyError:
                pass

    abridged_dataset['read_me'] = 'this is an abridged dataset with distortion data removed'

    with open(Path(ROOT_DIR, target_directory, STANDARD_DATASET_FILENAME), 'w') as f:
        json.dump(abridged_dataset, f)


if __name__ == '__main__':

    _parent_directory = r'/home/acb6595/coco/datasets/demo'

    # _sub_dirs = ['res', 'blur', 'noise']
    # _transfer_abridge_dataset = True
    #
    # for _sub_dir in _sub_dirs:
    #     _directory = Path(_parent_directory, _sub_dir)
    #     if _transfer_abridge_dataset:
    #         abridged_dataset_transfer(_parent_directory, _directory)
    #     illustrate(_directory)

    _directories = list(Path(_parent_directory).iterdir())

    _sub_directories = [_directory.parts[-1] for _directory in _directories]
    _sub_directories = [_sub_dir for _sub_dir in _sub_directories if str(_sub_dir[:4]).isnumeric()]
    _sub_directories = [_sub_dir for _sub_dir in _sub_directories if int(_sub_dir[:4]) >= 7]

    _path_extension = 'images/test'

    for _sub_dir in _sub_directories:

        _directory = Path(_parent_directory, _sub_dir, _path_extension)
        illustrate(_directory)

