import torch
import torch.nn as nn
import wandb
from src.d00_utils.definitions import PROJECT_ID, STANDARD_DATASET_FILENAME, ROOT_DIR, STANDARD_TEST_RESULT_FILENAME, \
    REL_PATHS
from src.d00_utils.functions import load_wandb_artifact_model, load_wandb_dataset_artifact, id_from_tags, get_config, \
    log_config
from src.d02_train.train import get_shard, run_shard, get_mean_accuracy, get_transform
from pathlib import Path
import argparse
import json

wandb.login()


def execute_test(model,
                 shard_ids,
                 data_abs_dir,
                 transform,
                 batch_size,
                 num_workers,
                 pin_memory,
                 device,
                 loss_func):
    running_loss = 0
    total_samples = 0
    weighted_accuracies = []
    shard_lengths = []
    shard_accuracies = []
    shard_performances = {}

    for shard_id in shard_ids:
        shard, shard_length = get_shard(shard_id, data_abs_dir, transform)

        model, shard_loss, shard_accuracy, labels, predicts = run_shard(model=model,
                                                                        train_eval_flag='eval',
                                                                        shard=shard,
                                                                        batch_size=batch_size,
                                                                        num_workers=num_workers,
                                                                        pin_memory=pin_memory,
                                                                        device=device,
                                                                        loss_func=loss_func,
                                                                        optimizer=None)

        shard_accuracies.append(shard_accuracy)
        shard_lengths.append(shard_length)

        labels = list(map(float, labels))  # convert torch tensor elements to floating point for json serialization
        predicts = list(map(float, predicts))

        shard_performances[shard_id] = {
            'labels': labels,
            'predicts': predicts
        }

        total_samples += shard_length

        scaled_loss = shard_loss / shard_length
        running_loss += scaled_loss
        weighted_accuracy = shard_length * shard_accuracy
        weighted_accuracies.append(weighted_accuracy)

    mean_accuracy = get_mean_accuracy(weighted_accuracies, total_samples)

    return running_loss, mean_accuracy, shard_accuracies, shard_lengths, shard_performances


def test_model(config):

    with wandb.init(project=PROJECT_ID, job_type='test_model', notes=config['description'], config=config) as run:

        config = wandb.config

        dataset_artifact_id = f"{config['test_dataset_id']}:{config['test_dataset_artifact_alias']}"
        model_artifact_id = f"{config['model_artifact_id']}:{config['model_artifact_alias']}"

        __, dataset = load_wandb_dataset_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)
        model = load_wandb_artifact_model(run, model_artifact_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        pin_memory = config['pin_memory']
        rgb_flag = config['rgb_flag']
        transform = get_transform(distortion_tags=[], rgb=rgb_flag)
        loss_function = getattr(nn, config['loss_func'])()
        artifact_type = 'test_result'

        test_shard_ids = dataset['test']['image_and_label_filenames']
        dataset_rel_dir = dataset['dataset_rel_dir']
        dataset_abs_dir = Path(ROOT_DIR, dataset_rel_dir)

        loss, accuracy, shard_accuracies, shard_lengths, shard_performances = execute_test(model,
                                                                                           test_shard_ids,
                                                                                           dataset_abs_dir,
                                                                                           transform,
                                                                                           batch_size,
                                                                                           num_workers,
                                                                                           pin_memory,
                                                                                           device,
                                                                                           loss_function)

        test_result = {
            'loss': loss,
            'accuracy': accuracy,
            'shard_accuracies': shard_accuracies,
            'shard_lengths': shard_lengths,
            'shard_performances': shard_performances
        }

        # log top level metrics for easy access on wandb dashboard
        wandb.log({
            'loss': test_result['loss'],
            'accuracy': test_result['accuracy'],
        })

        test_result_id = id_from_tags(artifact_type, [config['model_artifact_id'], config['test_dataset_id']])

        test_result_rel_dir = Path(REL_PATHS[artifact_type], test_result_id)
        test_result_abs_dir = Path(ROOT_DIR, test_result_rel_dir)
        Path.mkdir(test_result_abs_dir)
        full_test_result_path = Path(test_result_abs_dir, STANDARD_TEST_RESULT_FILENAME)

        with open(full_test_result_path, 'w') as file:
            json.dump(test_result, file)

        config_log_path = log_config(test_result_abs_dir, dict(config), return_path=True)

        test_result_artifact = wandb.Artifact(
            test_result_id,
            type=artifact_type,
            metadata=dict(config),
            description=config['description'],
        )

        test_result_artifact.add_file(full_test_result_path)
        test_result_artifact.add_file(config_log_path)

        run.log_artifact(test_result_artifact)
        run.name = test_result_id


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    test_model(run_config)

