from src.d01_pre_processing.distort_coco_dataset import resize_bbox
from src.d01_pre_processing.distortions import r0_coco
from src.d05_obj_det_analysis.analysis_tools import add_bboxes
from src.d00_utils.detection_functions import xywh_to_xyxy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def make_test_image(shape, bboxes, background=1.0, box_fill=0.5):

    """
    Makes a test image where regions inside of specified bounding boxes filled.

    :param shape: tuple
    :param bboxes: list, with each bounding box specified in xywh format
    :param background: grayscale value (float)
    :param box_fill: grayscale value (float)
    :return: PIL image
    """

    img = np.ones(shape) * background

    for box in bboxes:
        x, y, w, h = box
        img[y: y + h, x: x + w, :] = box_fill

    img = np.asarray(255 * img, dtype=np.uint8)
    img = Image.fromarray(img)

    return img


def display_image_bboxes(img, bboxes, ax=None, show=False, frac=None):

    xyxy_bboxes = [xywh_to_xyxy(*bbox) for bbox in bboxes]
    bbox_labels = ['FN' for i in range(len(xyxy_bboxes))]

    if ax is None:
        fig, ax = plt.subplots()
        if frac is not None:
            ax.set_title(str(frac))

    ax.imshow(img)
    add_bboxes(xyxy_bboxes, ax, bbox_labels=bbox_labels)

    if show:
        plt.show()

    return ax


def check_resizing(img, bboxes, num_iters=10):

    for i in range(num_iters):

        img_out, __, __, res_frac = r0_coco(img)
        img_out = Image.fromarray(img_out)
        bboxes_out = [resize_bbox(res_frac, bbox) for bbox in bboxes]
        display_image_bboxes(img_out, bboxes_out, show=True, frac=res_frac)


if __name__ == '__main__':

    _bboxes = [[200, 125, 50, 75], [50, 30, 90, 110], [210, 50, 250, 90]]
    _shape = (256, 512, 3)
    _img = make_test_image(shape=_shape, bboxes=_bboxes)
    display_image_bboxes(_img, bboxes=_bboxes, show=True)
    check_resizing(_img, _bboxes)
