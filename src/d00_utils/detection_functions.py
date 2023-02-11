import torch
from src.d00_utils.definitions import YOLO_TO_ORIGINAL_PAPER_KEY_MAPPING

def get_image_ids(coco_instance_images):
    return [x['id'] for x in coco_instance_images]


def map_annotations(coco_annotations, image_ids):

    mapped_annotations = {}

    for image_id in image_ids:

        filtered_annotations = [x for x in coco_annotations if x['image_id'] == image_id]
        mapped_annotations[image_id] = filtered_annotations

    return mapped_annotations


def map_boxes_labels(annotations, image_ids):

    mapped_filtered_annotations = {}

    mapped_annotations = map_annotations(annotations, image_ids)

    for image_id, annotations in mapped_annotations.items():

        bboxes = []
        object_ids = []

        for image_annotation in annotations:

            x, y, width, height = image_annotation['bbox']
            bbox = xywh_to_xyxy(x, y, width, height)
            object_id = image_annotation['category_id']
            bboxes.append(bbox)
            object_ids.append(object_id)

        bboxes = torch.tensor(bboxes)
        object_ids = torch.tensor(object_ids)
        mapped_filtered_annotations[image_id] = {'boxes': bboxes, 'labels': object_ids}

    return mapped_filtered_annotations


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


def listify(torch_data_dict):

    listed_data_dict = {}

    for image_id, torch_image_data in torch_data_dict.items():

        image_data = {}

        for key, torch_data in torch_image_data.items():
            if type(torch_data) == torch.Tensor:
                list_data = torch_data.tolist()
            else:
                list_data = torch_data
            image_data[key] = list_data

        listed_data_dict[image_id] = image_data

    return listed_data_dict


def yolo_result_to_target_fmt(result):
    # yolo_fmt_labels = result.boxes.cls.tolist()
    # labels = map(yolo_fmt_labels, yolo_fmt_labels)
    return dict(boxes = result.boxes.xyxy, scores = result.boxes.conf, yolo_fmt_labels=result.boxes.cls)


def translate_yolo_to_original_label_fmt(output):

    yolo_fmt_labels = output['yolo_fmt_labels']
    labels = yolo_to_original_labels(yolo_fmt_labels)
    output['labels'] = labels
    output.pop('yolo_fmt_labels')

    return output

def yolo_to_original_labels(yolo_fmt_labels):
    labels = list(map(_yolo_to_original_coco_label, yolo_fmt_labels))
    return labels


def _yolo_to_original_coco_label(label):
    return YOLO_TO_ORIGINAL_PAPER_KEY_MAPPING[int(label)]


def coco_standard_to_yolo_labels(labels: list):
    pass