import torch


def get_image_ids(coco_instance_images):
    return [x['id'] for x in coco_instance_images]


def map_annotations(coco_annotations, image_ids):

    mapped_annotations = {}

    for image_id in image_ids:

        filtered_annotations = [x for x in coco_annotations if x['image_id'] == image_id]
        mapped_annotations[image_id] = filtered_annotations

    return mapped_annotations


def assign_new_image_id():
    pass


def xywh_to_xyxy(x, y, width, height):
    x_min = x
    x_max = x + width
    y_min = y
    y_max = y + height
    new_bbox = [x_min, y_min, x_max, y_max]
    return new_bbox


def background_annotation(image):
    """
    Returns an annotation labeling entire image as background
    """

    w, h = image.size
    # bbox = xywh_to_xyxy(0, 0, w, h)
    bbox = [0, 0, w, h]
    object_id = 0
    bboxes = [bbox]
    object_ids = [object_id]
    bboxes = torch.tensor(bboxes, dtype=torch.float32)
    object_ids = torch.tensor(object_ids, dtype=torch.int64)
    annotation = {'boxes': bboxes, 'labels': object_ids}

    return annotation

