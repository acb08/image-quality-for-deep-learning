import torch
import argparse
import json
import wandb
from ultralytics import YOLO
import src.d02_train.train_obj_det
# import src.d02_train.train_detection
from src.d00_utils.definitions import WANDB_PID, STANDARD_DATASET_FILENAME, ROOT_DIR, STANDARD_TEST_RESULT_FILENAME, \
    REL_PATHS
from src.d00_utils.functions import load_wandb_model_artifact, load_wandb_data_artifact, id_from_tags, get_config, \
    log_config, construct_artifact_id
from pathlib import Path
import src.d00_utils.detection_functions as coco_functions
from src.d02_train.train_obj_det import get_loader, evaluate



def test_detection_model(config):

    with wandb.init(project=WANDB_PID, job_type='test_detection_model', notes=config['description'], config=config) as run:

        # config = wandb.config

        # dataset_artifact_id = f"{config['test_dataset_id']}:{config['test_dataset_artifact_alias']}"
        dataset_artifact_id, dataset_artifact_stem = construct_artifact_id(
            config['test_dataset_id'], artifact_alias=config['test_dataset_artifact_alias'])

        # model_artifact_id = f"{config['model_artifact_id']}:{config['model_artifact_alias']}"
        model_artifact_id, model_artifact_stem = construct_artifact_id(
            config['model_artifact_id'], artifact_alias=config['model_artifact_alias'])

        model = load_wandb_model_artifact(run, model_artifact_id)
        yolo_mode = type(model) == YOLO

        __, dataset = load_wandb_data_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)
        detection_dataset = src.d02_train.train_obj_det.wandb_to_detection_dataset(dataset, yolo_fmt=yolo_mode)

        torch.no_grad()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: ', device)
        model.to(device)
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        pin_memory = config['pin_memory']
        
        loader = get_loader(detection_dataset, batch_size=batch_size, num_workers=num_workers)

        # crop_flag = config['crop_flag']
        # last_distortion_type_flag = config['last_distortion_type_flag']
        # last_distortion_type_flag = dataset['last_distortion_type_flag']
        # dataset_split_key = config['dataset_split_key']
        # transform = get_transform(distortion_tags=[], crop=crop_flag)
        # loss_function = getattr(nn, config['loss_func'])()
        status_interval = config['status_interval']
        artifact_type = 'test_result'

        # if not dataset_split_key:
        #     dataset_split_key = 'test'

        dataset_rel_dir = dataset['dataset_rel_dir']

        # if last_distortion_type_flag:
        #     dataset_rel_dir = Path(dataset_rel_dir, REL_PATHS[last_distortion_type_flag])
        dataset_abs_dir = Path(ROOT_DIR, dataset_rel_dir)



        outputs, targets = evaluate(model=model,
                                    data_loader=loader,
                                    device=device,
                                    yolo_mode=yolo_mode)

        outputs = coco_functions.listify(outputs)
        targets = coco_functions.listify(targets)

        test_result = {
            'outputs': outputs,
            'targets': targets
        }

        test_result.update(config)

        # log top level metrics for easy access on wandb dashboard
        # wandb.log({
        #     'loss': test_result['loss'],
        #     'accuracy': test_result['accuracy'],
        # })

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

        test_result_artifact.add_file(full_test_result_path)
        test_result_artifact.add_file(config_log_path)

        run.log_artifact(test_result_artifact)
        run.name = test_result_id


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='r_scan_yolov8n_local.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'test_configs_detection'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    test_detection_model(run_config)
