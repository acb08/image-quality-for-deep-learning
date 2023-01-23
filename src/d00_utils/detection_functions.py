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


