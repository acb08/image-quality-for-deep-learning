from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json


def load_instances(directory, filename):

    filename = 'instances_val2017.json'
    with open(Path(directory, filename), 'r') as file:
        instances = json.load(file)

    return instances


class COCO(Dataset):

    def __init__(self, image_directory, instances):
        self.image_directory = Path(image_directory)
        self.instances = instances
        self.images = self.instances['images']
        self.annotations = self.instances['annotations']
        self.image_ids = [x['id'] for x in self.images]
        self.mapped_annotations = self.map_annotations()

    def map_annotations(self):
        mapped_annotations = {}
        for image_id in self.image_ids:
            image_annotations = [x for x in self.annotations if x['image_id'] == image_id]
            bboxes = []
            object_ids = []
            for image_annotation in image_annotations:
                bbox = image_annotation['bbox']
                object_id = image_annotation['id']
                bboxes.append(bbox)
                object_ids.append(object_id)
            mapped_annotations[image_id] = {'boxes': bboxes, 'labels': object_ids}
        return mapped_annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_data = self.images[idx]
        file_name = image_data['filename']
        image_id = image_data['id']
        image = Image.open(Path(self.image_directory, file_name))

        image_annotations = self.mapped_annotations[image_id]
        return image, image_annotations


if __name__ == '__main__':
    _instance_dir = r'/home/acb6595/coco/annotations'
    _instance_filename = 'instances_val2017.json'
    _instances = load_instances(_instance_dir, _instance_filename)
    _coco = COCO(r'/home/acb6595/coco/val2017', _instances)
    _annotation_map = _coco.mapped_annotations
    print(len(_annotation_map))
