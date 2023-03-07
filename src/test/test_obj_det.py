import torch
import argparse
import json
import wandb
from ultralytics import YOLO
from src.utils.definitions import WANDB_PID, STANDARD_DATASET_FILENAME, ROOT_DIR, STANDARD_TEST_RESULT_FILENAME, \
    REL_PATHS
from src.utils.functions import load_wandb_model_artifact, load_wandb_data_artifact, id_from_tags, get_config, \
    log_config, construct_artifact_id, wandb_to_detection_dataset
from pathlib import Path
import src.utils.detection_functions as coco_functions
from src.train.rcnn import get_loader, evaluate


def test_detection_model(config):

    with wandb.init(project=WANDB_PID, job_type='test_detection_model', notes=config['description'], config=config) as run:

        dataset_artifact_id, dataset_artifact_stem = construct_artifact_id(
            config['test_dataset_id'], artifact_alias=config['test_dataset_artifact_alias'])

        model_artifact_id, model_artifact_stem = construct_artifact_id(
            config['model_artifact_id'], artifact_alias=config['model_artifact_alias'])

        model = load_wandb_model_artifact(run, model_artifact_id)
        yolo_mode = type(model) == YOLO

        __, dataset = load_wandb_data_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)
        detection_dataset = wandb_to_detection_dataset(dataset, yolo_fmt=yolo_mode)

        torch.no_grad()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: ', device)
        model.to(device)
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        
        loader = get_loader(detection_dataset, batch_size=batch_size, num_workers=num_workers)

        status_interval = config['status_interval']
        artifact_type = 'test_result'

        outputs, targets = evaluate(model=model,
                                    data_loader=loader,
                                    device=device,
                                    status_interval=status_interval,
                                    yolo_mode=yolo_mode)

        outputs = coco_functions.listify(outputs)
        targets = coco_functions.listify(targets)

        test_result = {
            'outputs': outputs,
            'targets': targets
        }

        test_result.update(config)

        test_result_id = id_from_tags(artifact_type, [model_artifact_stem, dataset_artifact_stem])

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

        test_result_artifact.add_file(str(full_test_result_path))
        test_result_artifact.add_file(str(config_log_path))

        run.log_artifact(test_result_artifact)
        run.name = test_result_id


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='n_scan_yolov8n-noise_local.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'test_configs_detection'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    test_detection_model(run_config)
