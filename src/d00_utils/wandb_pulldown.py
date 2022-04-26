import wandb
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME
from src.d00_utils.functions import read_json_artifact

if __name__ == '__main__':

    run = wandb.init()
    artifact = run.use_artifact('austinbergstrom/places365/train_256_standard:v1', type='train_dataset')
    artifact_dir = artifact.download()
    artifact_filename = STANDARD_DATASET_FILENAME

    dataset = read_json_artifact(artifact_dir, artifact_filename)
