"""
Takes a run config file and a destination directory and transfers the necessary pieces of the project to the new
directory.
"""
import argparse
from src.utils._preload_tools import get_dataset_dir, duplicate_root, duplicate_project_sub_dirs, transfer_dataset, \
    transfer_code, transfer_project_config
from src.utils.definitions import YOLO_TRAIN_CONFIG_DIR, RCNN_TRAIN_CONFIG_DIR, TEST_DETECTION_CONFIG_DIR
from src.utils.functions import get_config
from pathlib import Path


RUN_TO_ARTIFACT_TYPE = {
    'train': 'train_dataset',
    'train_rcnn': 'train_dataset',
    'test': 'test_dataset',
}

RUN_TO_CONFIG_DIR = {
    'train': YOLO_TRAIN_CONFIG_DIR,
    'train_rcnn': RCNN_TRAIN_CONFIG_DIR,
    'test': TEST_DETECTION_CONFIG_DIR
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='train_config.yml', help='config filename to be used')
    parser.add_argument('--run_type',
                        default='train',
                        help='train, train_rcnn, test')
    parser.add_argument('--destination', default='~/scratch')
    args_parsed = parser.parse_args()

    run_type = args_parsed.run_type
    dataset_artifact_type = RUN_TO_ARTIFACT_TYPE[run_type]

    config_dir = RUN_TO_CONFIG_DIR[run_type]

    run_config = get_config(args=None,
                            config_dir=config_dir,
                            config_name=args_parsed.config_name)

    # artifact_id_key = 'train_dataset_id' if 'train_dataset_id' in run_config else 'test_dataset_id'

    destination = Path(args_parsed.destination).expanduser()
    destination_root = duplicate_root(destination=destination)
    dataset_starting_path = get_dataset_dir(run_config, artifact_type=dataset_artifact_type)

    duplicate_project_sub_dirs(destination_root)
    transfer_code(destination_root=destination_root)
    transfer_dataset(dataset_starting_path=dataset_starting_path, destination_root=destination_root,
                     dataset_artifact_type=dataset_artifact_type)
    transfer_project_config(destination_root=destination_root)
