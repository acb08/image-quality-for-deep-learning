"""
Used to pre-load a wandb artifact before the artifact is actually used by a subsequent script.  Specifically, this
script should load a wandb artifact based on a dataset distortion, training, or testing run config BEFORE the dataset
distortion, training, or testing script is actually run.

Use case: most wandb artifacts here are just metadata, with the actual data (images) stored locally.  To use the faster
scratch storage in the SPORC, data needs to be transferred to the scratch space at run time.  This script gets the
relevant metadata from the artifact and returns a path to the actual dataset, which the calling script can use to
find the data to be transferred.
"""
from src.utils._preload_tools import get_dataset_dir, log_artifact_path
from src.utils.definitions import YOLO_TRAIN_CONFIG_DIR
from src.utils.functions import get_config
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='train_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=YOLO_TRAIN_CONFIG_DIR,
                        help="configuration file directory")
    parser.add_argument('--metadata_dir', default='./')
    parser.add_argument('--metadata_filename', default='dataset_path.txt')
    args_parsed = parser.parse_args()

    run_config = get_config(args_parsed)
    dataset_path = get_dataset_dir(run_config, artifact_type='train_dataset')
    log_artifact_path(path=dataset_path,
                      args=args_parsed)
