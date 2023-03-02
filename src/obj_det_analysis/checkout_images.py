import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
from pathlib import Path


def load_instances():

    directory = r'/home/acb6595/coco/annotations'
    filename = 'instances_val2017.json'
    with open(Path(directory, filename), 'r') as file:
        instances = json.load(file)

    return instances


def checkout_image(index, instances):

    image_directory = r'/home/acb6595/coco/val2017'
    image_info = instances['images'][index]
    filename = image_info['file_name']
    image = np.asarray(Image.open(Path(image_directory, filename)))

    plt.figure()
    plt.imshow(image)
    plt.title(image_info['id'])
    plt.show()


def all_category_labels(instances):
    categories = instances['categories']
    ids = []
    for cat_dict in categories:
        cat_id = cat_dict['id']
        ids.append(cat_id)

    print(ids)
    print(np.min(ids))


if __name__ == '__main__':

    _instances = load_instances()
    all_category_labels(_instances)
    checkout_image(2200, _instances)
