"""
Band-aide script to restore a WANDB artifact accidentally deleted from wandb servers but still available locally
"""
import wandb
import argparse
from src.d00_utils.functions import get_config, load_wandb_data_artifact, id_from_tags
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, WANDB_PID, ROOT_DIR
from pathlib import Path


def relog_dataset(config, restore_name, restore_artifact_path, restore_artifact_parent_artifact_name):

    parent_dataset_id = config['parent_dataset_id']
    parent_artifact_alias = config['parent_artifact_alias']
    parent_artifact_filename = config['parent_artifact_filename']
    if parent_artifact_filename == 'standard':
        parent_artifact_filename = STANDARD_DATASET_FILENAME
    artifact_type = config['artifact_type']
    num_images = config['num_images']
    images_per_file = config['images_per_file']
    datatype_key = config['datatype_key']
    artifact_filename = config['artifact_filename']
    if artifact_filename == 'standard':
        artifact_filename = STANDARD_DATASET_FILENAME
    distortion_tags = config['distortion_tags']
    distortion_type_flags = config['distortion_type_flags']
    dataset_split_key = config['dataset_split_key']
    iterations = config['iterations']
    description = config['description']

    with wandb.init(project=WANDB_PID, job_type='distort_dataset', tags=distortion_tags, notes=description,
                    config=config) as run:

        # include to re-build wandb artifact graph
        parent_artifact, parent_dataset = load_wandb_data_artifact(run,
                                                                   restore_artifact_parent_artifact_name,
                                                                   STANDARD_DATASET_FILENAME)
        if 'name_string' in config.keys() and config['name_string'] is not None:
            name_string = config['name_string']
            new_dataset_id, new_dataset_rel_dir = id_from_tags(artifact_type, [name_string], return_dir=True)

        run_metadata = config
        run_metadata_additions = {
            'artifact_filename': artifact_filename,
            'root_dir_at_run': str(ROOT_DIR),
            'dataset_rel_dir': str(new_dataset_rel_dir)
        }
        run_metadata.update(run_metadata_additions)

        new_artifact_id = restore_name
        new_artifact_path = restore_artifact_path

        new_artifact = wandb.Artifact(new_artifact_id,
                                      type=artifact_type,
                                      metadata=run_metadata)
        new_artifact.add_file(new_artifact_path)
        new_artifact.metadata = run_metadata
        run.log_artifact(new_artifact)


if __name__ == '__main__':

    config_filename = 'pl_fr.yml'  # set manually here but use argparse and get_config() to maintain commonality

    artifact_restore_name = '0008-tst-full_range_mega_set_2_noise'
    _restore_parent_artifact_name = '0008-tst-full_range_mega_set_2_blur:latest'
    _restore_artifact_path = '/home/acb6595/places/datasets/test/0008-tst-full_range_mega_set_2/3-noise/dataset.json'
    _restore_artifact_name = '0008-tst-full_range_mega_set_2_noise'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    relog_dataset(run_config, _restore_artifact_name, _restore_artifact_path, _restore_parent_artifact_name)
