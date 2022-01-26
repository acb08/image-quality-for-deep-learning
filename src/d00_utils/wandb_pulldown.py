import json
import wandb
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME
from src.d00_utils.functions import read_artifact

if __name__ == '__main__':

    run = wandb.init()
    artifact = run.use_artifact('austinbergstrom/places_dry_run/0016_pan:v0', type='test_dataset')
    artifact_dir = artifact.download()
    artifact_filename = STANDARD_DATASET_FILENAME

    dataset = read_artifact(artifact_dir, artifact_filename)
