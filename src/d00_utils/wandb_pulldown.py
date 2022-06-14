import wandb
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, STANDARD_TEST_RESULT_FILENAME, PROJECT_ID
from src.d00_utils.functions import read_json_artifact, load_wandb_data_artifact
from pathlib import Path

if __name__ == '__main__':

    # run = wandb.init(project='sat6_v2')
    # artifact = run.use_artifact('0017-rlt-0005-resnet50_sat6-full_range_best_loss-0008-tst-r_fr_s6-b_fr_s6-n_fr_s6_noise:v0000')
    # artifact_dir = artifact.download()
    # artifact_filename = STANDARD_TEST_RESULT_FILENAME
    #
    # dataset = read_json_artifact(artifact_dir, artifact_filename)
    #

    artifact_id = 'austinbergstrom/sat6_v2/0011-tst-endpoint_noise:v0'

    with wandb.init(project=PROJECT_ID, job_type='load_artifact') as run:
        artifact_dir, dataset = load_wandb_data_artifact(run, artifact_id, STANDARD_DATASET_FILENAME)

