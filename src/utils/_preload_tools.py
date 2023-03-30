from pathlib import Path

import wandb

from src.utils.definitions import WANDB_PID, STANDARD_DATASET_FILENAME, ROOT_DIR, REL_PATHS
from src.utils.functions import construct_artifact_id, load_wandb_data_artifact


def get_dataset_dir(config, artifact_type):

    with wandb.init(project=WANDB_PID, job_type='artifact_preload', config=config) as run:

        dataset_artifact_id, __ = construct_artifact_id(config['train_dataset_id'],
                                                        artifact_alias=config['train_dataset_artifact_alias'])
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
