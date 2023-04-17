"""
Sanity check for resolution distortion functions.
"""
import json
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
from src.utils.definitions import ROOT_DIR, REL_PATHS


def get_image_filenames(directory):
    filenames = list(Path(directory).iterdir())
    filenames = [file for file in filenames if Path(file).is_file()]
    filenames = [file for file in filenames if Path(file).suffix in {'.png', '.jpg'}]
    return filenames


def get_image_sizes(directory, image_filenames=None):

    if image_filenames is None:
        image_filenames = get_image_filenames(directory)

    image_sizes = {}
    totals = np.zeros(2)
    counter = 0

    for filename in image_filenames:

        img = Image.open(Path(directory, filename))
        size = img.size
        totals += np.asarray(size)
        counter += 1

        image_sizes[str(filename)] = list(size)

    mean_size = totals / counter
    print('mean image size: ', mean_size)

    return image_sizes


def log_image_sizes(output_dir, image_sizes, filename):
    with open(Path(output_dir, filename), 'w') as f:
        json.dump(image_sizes, f)


if __name__ == '__main__':

    # _directory = r'~/coco/datasets/train/0023trn-coco_fr_train/images/train'
    _directory = r'/home/acb6595/coco/datasets/train/coco128/images/train2017'
    _directory = Path(_directory).expanduser()
    _output_dir = '~/coco/misc'
    _output_dir = Path(_output_dir).expanduser()

    _images_sizes = get_image_sizes(_directory)
    log_image_sizes(_output_dir, _images_sizes, 'image_sizes_0023trn-coco_fr_train.json')

