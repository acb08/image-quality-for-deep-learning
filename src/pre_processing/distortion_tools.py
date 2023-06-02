import copy
import numpy as np
from PIL import Image
from src.utils.definitions import WELL_DEPTH

FRAGILE_ANNOTATION_KEYS = ['area', 'segmentation']


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


def resize_bbox(res_frac, bbox):

    bbox = np.asarray(bbox)
    bbox = res_frac * bbox
    bbox = list(bbox)

    return bbox


def image_to_electrons(image, well_depth=WELL_DEPTH):

    # assert type(image) == Image.Image

    image = np.asarray(image, dtype=np.uint8)
    image = normalize(image)
    electrons = image * well_depth
    # electrons = np.asarray(electrons, dtype=np.int32)

    return electrons


def electrons_to_image(electrons, well_depth=WELL_DEPTH):

    image = electrons / well_depth
    image = convert_to_uint8(image)
    image = Image.fromarray(image)

    return image


def convert_to_uint8(img):

    assert np.min(img) >= 0.0
    assert np.max(img) <= 1.0

    img = (2 ** 8 - 1) * img
    img = np.asarray(img, dtype=np.uint8)

    return img


def normalize(img):

    assert np.max(img) <= 255
    assert np.min(img) >= 0

    return img / (2 ** 8 - 1)


def apply_partial_poisson_distribution(signal, dc_fraction=0):

    dc_component = dc_fraction * signal
    poisson_component = signal - dc_component

    return dc_component + np.random.poisson(poisson_component)


def apply_poisson_distribution(signal):
    return np.random.poisson(signal)
