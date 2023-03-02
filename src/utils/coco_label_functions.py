from yaml import safe_load, safe_dump
from src.utils.definitions import OBSOLETE_CLASS_IDS, ROOT_DIR, REL_PATHS, \
    STANDARD_TO_ORIGINAL_PAPER_LABEL_MAP_FILENAME, YOLO_TO_ORIGINAL_PAPER_LABEL_MAP_FILENAME
from pathlib import Path


YOLO_LABEL_FILENAME = 'coco_labels_yolo.yml'
STANDARD_LABEL_FILENAME = 'coco_labels.txt'
COCO_LABELS_ORIGINAL_PAPER_FILENAME = 'coco_labels_original_paper.txt'
BACKGROUND = {0: '__background__'}

def parse_label_file_txt(verbose=True):

    name_label_map = {}

    with open(Path(ROOT_DIR, REL_PATHS['label_definition_files'], STANDARD_LABEL_FILENAME), 'r') as f:

        for i, line in enumerate(f):

            name = line.split("'")[-2]
            label = line.split(":")[0].replace('{', '')
            name_label_map[int(label)] = name

            if verbose:
                print(f'{label}: {name}')

    return name_label_map


def get_yolo_labels():

    with open(Path(ROOT_DIR,  REL_PATHS['label_definition_files'], YOLO_LABEL_FILENAME), 'r') as f:
        name_label_map = safe_load(f)

    return name_label_map['names']


def convert_yolo_to_standard_label_keys(yolo_labels):
    converted_labels = {key + 1: val for key, val in yolo_labels.items()}
    converted_labels.update(BACKGROUND)
    return converted_labels

def compare(standard_labels, yolo_labels, key_offset=1):

    yolo_labels = yolo_labels['names']

    full_agreement = True

    for yolo_label, yolo_name in yolo_labels.items():
        yolo_label = int(yolo_label)
        standard_compare_label = yolo_label + key_offset
        standard_name = standard_labels[standard_compare_label]
        if yolo_name != standard_name:
            print(yolo_name, standard_name)
            full_agreement = False

    print('agreement after offset correction: ', full_agreement)


def check_label_set(image_mapped_detection_data):

    all_labels = set()

    for image_id, data in image_mapped_detection_data.items():
        labels = set(data['labels'])
        all_labels.update(labels)

    obsolete_labels_present = all_labels.intersection(OBSOLETE_CLASS_IDS)

    return all_labels, obsolete_labels_present


def get_original_paper_labels():

    original_labels = {}

    with open(Path(ROOT_DIR,  REL_PATHS['label_definition_files'], COCO_LABELS_ORIGINAL_PAPER_FILENAME), 'r') as f:
        for i, line in enumerate(f):
            label = line.rstrip('\n')
            original_labels[i] = label

    return original_labels


def reverse_keys_values(data):
    return {val: key for key, val in data.items()}

def check_consistency(starting_dict, target_dict, key_mapping):

    starting_dict_mapped = apply_key_map(key_mapping, starting_dict)

    for key_mapped, val_mapped in starting_dict_mapped.items():
        assert val_mapped == target_dict[key_mapped]



def log_key_map(standard_to_original_key_map, filename, double_check=False):

    with open(Path(ROOT_DIR, REL_PATHS['project_config'], filename), 'w') as f:
        safe_dump(standard_to_original_key_map, f)

    if double_check:
        with open(Path(ROOT_DIR, REL_PATHS['project_config'], filename), 'r') as f:
            data = safe_load(f)
        return data


def convert_yolo_to_original_label_keys(yolo_labels, yolo_to_standard_mapping, standard_to_original_mapping):
    pass


def map_equivalent_dict_keys(starting_keyed_dict, target_keyed_dict):
    """
    Provides a map between the keys in starting_keyed_dict to target_keyed_dict, under the condition that the
    dictionaries have identical values. Returns key mapping starting_to_target_key_map such
    starting_keyed_dict[starting_key] = target_keyed_dict[starting_to_target_key_map[starting_key]]
    """

    starting_key_reverse_mapping = reverse_keys_values(starting_keyed_dict)
    target_key_reverse_mapping = reverse_keys_values(target_keyed_dict)

    starting_to_target_key_map = {standard_key: target_key_reverse_mapping[label] for
                                    label, standard_key in starting_key_reverse_mapping.items()}

    for starting_key in starting_keyed_dict.keys():
        assert starting_keyed_dict[starting_key] == target_keyed_dict[starting_to_target_key_map[starting_key]]

    return starting_to_target_key_map


def apply_key_map(key_map, starting_dict):
    return {key_map[starting_key]: starting_val for starting_key, starting_val in starting_dict.items()}


def log_yolo_to_original_mapping():

    original_paper_labels = get_original_paper_labels()
    yolo_labels = get_yolo_labels()
    yolo_to_original_key_map = map_equivalent_dict_keys(yolo_labels, original_paper_labels)
    check_consistency(starting_dict=yolo_labels,
                      target_dict=original_paper_labels,
                      key_mapping=yolo_to_original_key_map)
    log_key_map(yolo_to_original_key_map,
                filename=YOLO_TO_ORIGINAL_PAPER_LABEL_MAP_FILENAME,
                double_check=False)


if __name__ == '__main__':

    log_yolo_to_original_mapping()
