from pathlib import Path

import wandb

from src.utils.definitions import WANDB_PID, STANDARD_DATASET_FILENAME, ROOT_DIR, REL_PATHS, CODE_ROOT
from src.utils.functions import construct_artifact_id, load_wandb_data_artifact
import shutil
import os

DIRECTORY_KEYS = [
    'train_dataset',
    'test_dataset',
    'model',
    'test_result',
    'analysis',
    'project_config'
]

_DATASET_ID_KEYS = {
    'train_dataset': 'train_dataset_id',
    'test_dataset': 'test_dataset_id'
}


def _get_dataset_artifact_alias(config, ending='dataset_artifact_alias'):

    for key in config.keys():
        if key[-len(ending):] == ending:
            return config[key]

    return 'latest'


def get_dataset_dir(config, artifact_type):

    with wandb.init(project=WANDB_PID, job_type='artifact_preload', config=config) as run:

        dataset_id_key = _DATASET_ID_KEYS[artifact_type]
        artifact_alias = _get_dataset_artifact_alias(config)
        dataset_artifact_id, __ = construct_artifact_id(config[dataset_id_key],
                                                        artifact_alias=artifact_alias)
        __, dataset = load_wandb_data_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)
        rel_path = dataset['dataset_rel_dir']
        abs_path = Path(ROOT_DIR, rel_path)
        stem_directory = get_artifact_stem_directory(path=abs_path,
                                                     artifact_type=artifact_type)

    return stem_directory


def get_artifact_stem_directory(path, artifact_type):

    """
    In most cases, the dataset path stored in a dataset artifact points to the directory containing images, which is
    generally one or more levels deep within the relevant directory structure unique to a particular dataset. The
    function finds the top absolute path to the top level directory unique to a particular dataset.
    """

    assert artifact_type in {'train_dataset', 'test_dataset', 'model'}

    artifact_parent_dir = Path(ROOT_DIR, REL_PATHS[artifact_type])
    sub_path = Path(path).relative_to(artifact_parent_dir)
    artifact_stem = sub_path.parts[0]
    artifact_stem = Path(artifact_parent_dir, artifact_stem)

    print(f'artifact_stem: {artifact_stem}')

    return artifact_stem


def duplicate_root(destination, parents_ok=False):

    root_local = Path(ROOT_DIR).parts[-1]
    destination_root = Path(destination, root_local)
    if not destination_root.is_dir():
        Path.mkdir(destination_root, parents=parents_ok)

    return destination_root


def duplicate_project_sub_dirs(destination_root):
    for dir_key in DIRECTORY_KEYS:
        target_directory = Path(destination_root, REL_PATHS[dir_key])
        if not target_directory.is_dir():
            Path.mkdir(target_directory, parents=True, exist_ok=True)


def _copy_directory(source, destination):
    command_string = f'rsync -azh {source} {destination}'
    os.system(command_string)


def transfer_dataset(dataset_starting_path, destination_root, dataset_artifact_type):
    destination_rel_path = REL_PATHS[dataset_artifact_type]
    destination = Path(destination_root, destination_rel_path)
    return _copy_directory(source=dataset_starting_path, destination=destination)


def transfer_code(destination_root):
    return _copy_directory(source=CODE_ROOT, destination=destination_root)


def transfer_project_config(destination_root):
    project_config_dir = Path(ROOT_DIR, REL_PATHS['project_config'])
    return _copy_directory(source=project_config_dir, destination=destination_root)


def log_artifact_path(path, directory=None, filename=None, args=None):

    if args is not None:
        directory = args.metadata_dir
        filename = args.metadata_filename

    directory = Path(directory)
    if not directory.is_dir():
        Path.mkdir(directory)

    output_path = Path(directory, filename)

    with open(output_path, 'w') as f:
        f.write(str(path))
        f.write('\n')
