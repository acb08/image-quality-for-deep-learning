import wandb
import json
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, ORIGINAL_DATASETS, PROJECT_ID, STANDARD_DATASET_FILENAME
from src.d00_utils.functions import load_original_dataset, log_metadata

"""
Logs undistorted datasets as a W&B artifact and puts metadata into .json format 
"""

wandb.login()


def main(dataset_id, description=None):
    """
    logs an unmodified dataset as a W&B artifact. Creates a dataset.json file to store images names and labels
    in a list format
    :param dataset_id: pointer to existing dataset
    :param description: free form description of the artifact to be logged

    """

    # info on the original, un-logged dataset
    dataset_info = ORIGINAL_DATASETS[dataset_id]
    dataset_rel_path = dataset_info['rel_path']  # image directory
    artifact_type = dataset_info['artifact_type']

    # paths for standardizing with other artifacts
    dataset_filename = STANDARD_DATASET_FILENAME
    artifact_rel_path = Path(dataset_rel_path, dataset_filename)
    artifact_path = Path(ROOT_DIR, artifact_rel_path)

    run_metadata = {
        'parent_dataset_id': dataset_id,
        'description': description,
        'dataset_rel_dir': str(dataset_rel_path),
        'artifact_filename': dataset_filename,  # used for reading file after W&B directory download
        'distortion_tags': [],
        'distortion_iterations': [],
        'ROOT_DIR_at_run': str(ROOT_DIR),
    }

    log_metadata(artifact_type, dataset_id, run_metadata)  # logs in local json file

    with wandb.init(project=PROJECT_ID, job_type='load_dataset') as run:

        image_metadata = load_original_dataset(dataset_id)
        image_metadata.update(run_metadata)
        dataset = wandb.Artifact(
            dataset_id,
            type=artifact_type,
            metadata=run_metadata,
            description=description,
        )

        with open(artifact_path, 'w') as file:
            json.dump(image_metadata, file)

        dataset.add_file(artifact_path)
        run.log_artifact(dataset)
        wandb.finish()


if __name__ == '__main__':

    _description = 'test of ability to log Places365 train_256 as artifact. parent_dataset_id is ' \
                   'identical to dataset_id when logging initial datasets'
    main('train_256_standard', description=_description)
