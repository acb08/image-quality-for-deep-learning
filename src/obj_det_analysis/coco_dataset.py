from pathlib import Path
import json
# from src.utils.classes import COCO
# from src.utils import definitions
from src.train import rcnn as run


def load_instances(directory, filename):

    with open(Path(directory, filename), 'r') as file:
        instances = json.load(file)

    return instances


if __name__ == '__main__':
    # _instance_dir = r'/home/acb6595/coco/annotations'
    # _instance_filename = 'instances_val2017.json'
    # _instances = load_instances(_instance_dir, _instance_filename)
    # _coco = COCO(r'/home/acb6595/coco/val2017', _instances)
    _coco = run.get_dataset(cutoff=2)
    _annotation_map = _coco.mapped_boxes_labels
    print(len(_annotation_map))
