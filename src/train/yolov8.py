import copy
import yaml
import wandb
import argparse
import shutil
from pathlib import Path
from src.utils.coco_label_functions import get_yolo_labels
from src.utils.functions import get_config
from src.utils.definitions import STANDARD_DATASET_FILENAME, ROOT_DIR, WANDB_PID, REL_PATHS, HOST, YOLO_TRAIN_CONFIG_DIR
from src.utils.functions import construct_artifact_id, load_wandb_data_artifact, load_wandb_model_artifact, \
    id_from_tags, log_model_helper


def yaml_on_the_fly(rel_path=None,
                    train=None,
                    val=None,
                    test=None,
                    path=None):

    """

    :param rel_path: relative path from project's ROOT_DIR to yolo_root containing images and labels directories
    :param train: from rel_path to train images
    :param val: from rel_path to validation images
    :param test:  from rel_path to test images
    :param path: alternate (alias) for rel_path (cannot both be passed)
    :return: path to temporary yaml file
    """

    if rel_path is not None and path is not None:
        raise ValueError

    if rel_path is None:
        rel_path = path

    path = str(Path(ROOT_DIR, rel_path))
    train = str(train)
    val = str(val)

    names = get_yolo_labels()

    data = dict(
        path=path,
        train=train,
        val=val,
        test=test,
        names=names,
    )

    save_dir = Path(ROOT_DIR, REL_PATHS['temp_yaml'])
    if not save_dir.is_dir():
        Path.mkdir(save_dir)
    save_path = Path(save_dir, '_data.yaml')
    with open(save_path, 'w') as f:
        yaml.safe_dump(data, f)

    return str(save_path)


def count_relevant_sub_dirs(parent_dir, stem):

    num_chars = len(stem)
    contents = list(Path(parent_dir).iterdir())
    sub_dirs = [item for item in contents if item.is_dir()]
    counted_sub_dirs = [sub_dir for sub_dir in sub_dirs if sub_dir.stem[:num_chars] == stem]
    num_counted_sub_dirs = len(counted_sub_dirs)

    highest_dir_idx = check_highest_dir_idx(sub_dirs, stem)

    return num_counted_sub_dirs, highest_dir_idx


def check_highest_dir_idx(sub_dirs, stem):

    num_chars = len(stem)
    idx_strings = [sub_dir.stem[num_chars:] for sub_dir in sub_dirs]
    indices = [int(idx_string) for idx_string in idx_strings if idx_string.isdigit()]
    if len(indices) > 0:
        return max(indices)
    else:
        return None


def get_yolo_dir_name(stem, dir_count):
    if dir_count == 0:
        return stem
    else:
        return f'{stem}{dir_count + 1}'


def get_yolo_output_dirs(parent):

    parent = Path(parent)

    default_train_sub_dir = REL_PATHS['yolo_train_default_output_subdir']
    default_val_sub_dir = REL_PATHS['yolo_val_default_output_subdir']

    train_sub_dir_count, high_train_dir_idx = count_relevant_sub_dirs(parent, stem=default_train_sub_dir)
    train_sub_dir_name = get_yolo_dir_name(default_train_sub_dir, train_sub_dir_count)

    if high_train_dir_idx is not None:
        assert high_train_dir_idx == train_sub_dir_count

    val_sub_dir_count, high_val_dir_idx = count_relevant_sub_dirs(parent, stem=default_val_sub_dir)
    val_sub_dir_name = get_yolo_dir_name(default_val_sub_dir, val_sub_dir_count)
    if high_val_dir_idx is not None:
        assert high_val_dir_idx == val_sub_dir_count

    return train_sub_dir_name, val_sub_dir_name


def get_yolo_weight_paths(train_sub_dir):
    return Path(train_sub_dir, REL_PATHS['yolo_best_weights']), Path(train_sub_dir, REL_PATHS['yolo_last_weights'])


def load_tune_model(config):

    run_tags = copy.deepcopy(config['distortion_tags'])

    if 'device' in config.keys():
        device = config['device']
    else:
        device = 0

    wandb.login()

    with wandb.init(project=WANDB_PID, job_type='train_model', config=config, tags=run_tags) as run:

        config = wandb.config  # allows wandb parameter sweeps (not currently implemented)

        dataset_artifact_id, __ = construct_artifact_id(config['train_dataset_id'],
                                                        artifact_alias=config['train_dataset_artifact_alias'])
        starting_model_artifact_id, __ = construct_artifact_id(config['starting_model_id'],
                                                               artifact_alias=config['starting_model_artifact_alias'])

        model, arch, __ = load_wandb_model_artifact(run, starting_model_artifact_id, return_configs=True)

        config['arch'] = arch
        config['ROOT_DIR_at_run'] = str(ROOT_DIR)

        __, dataset = load_wandb_data_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)

        yolo_cfg = dataset['yolo_cfg']
        temp_yaml_cfg_path = yaml_on_the_fly(**yolo_cfg)

        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        # num_workers = config['num_workers']

        description = config['description']
        artifact_type = config['artifact_type']

        model_id_tags = [arch]

        if 'name_string' in config.keys() and config['name_string'] is not None:
            model_id_tags.append(config['name_string'])
            new_model_id = id_from_tags(artifact_type, model_id_tags)

        else:
            new_model_id = id_from_tags(artifact_type, model_id_tags)

        new_model_rel_dir = Path(REL_PATHS[artifact_type], new_model_id)
        output_dir = Path(ROOT_DIR, new_model_rel_dir)
        output_dir.mkdir(parents=True)

        # path shenanigans there because yolo models auto-create output directories, and I have not found a good way
        # to just access them as a class attribute, so the code below attempts to replicate what yolo is doing
        # internally until I come across a simple way to access the directory info directly
        train_sub_dir, val_sub_dir = get_yolo_output_dirs(parent=output_dir)
        train_sub_dir = Path(output_dir, train_sub_dir)

        best_weights_path, last_epoch_weights_path = get_yolo_weight_paths(train_sub_dir)
        best_weights_dir = best_weights_path.parent
        best_weights_rel_dir = best_weights_dir.relative_to(ROOT_DIR)

        best_weights_filename = best_weights_path.name
        # model_metadata = dict(config)
        # model_metadata['model_filename'] = best_weights_filename

        metadata = dict(config)

        best_loss_model_file_config = {
            'model_rel_dir': str(best_weights_rel_dir),
            'model_filename': best_weights_filename
        }
        metadata['model_file_config'] = best_loss_model_file_config

        model.train(data=temp_yaml_cfg_path,
                    epochs=num_epochs,
                    batch=batch_size,
                    project=output_dir,
                    device=device)

        model_helper_path = log_model_helper(best_weights_path.parent, metadata)

        best_loss_model_artifact = wandb.Artifact(
            f'{new_model_id}_best_loss',
            type=artifact_type,
            metadata=metadata,
            description=description
        )

        best_loss_model_artifact.add_file(str(best_weights_path))
        best_loss_model_artifact.add_file(str(model_helper_path))
        additional_files = get_target_file_paths(train_sub_dir)
        add_em_all(best_loss_model_artifact, additional_files)
        run.log_artifact(best_loss_model_artifact)

        last_epoch_weights_destination_dir = Path(last_epoch_weights_path.parent, 'last')
        last_epoch_weights_destination_dir.mkdir()
        last_epoch_rel_dir = last_epoch_weights_destination_dir.relative_to(ROOT_DIR)
        last_epoch_model_filename = last_epoch_weights_path.name
        last_epoch_model_path = Path(last_epoch_weights_destination_dir, last_epoch_model_filename)
        shutil.move(last_epoch_weights_path, last_epoch_model_path)

        last_epoch_model_file_config = {
            'model_rel_dir': str(last_epoch_rel_dir),
            'model_filename': last_epoch_model_filename
        }
        last_epoch_metadata = copy.deepcopy(metadata)
        last_epoch_metadata['model_file_config'] = last_epoch_model_file_config

        last_epoch_helper_path = log_model_helper(last_epoch_weights_destination_dir, last_epoch_metadata)
        last_epoch_model_artifact = wandb.Artifact(
            f'{new_model_id}_last_epoch',
            type=artifact_type,
            metadata=last_epoch_metadata,
            description=description
        )
        last_epoch_model_artifact.add_file(str(last_epoch_model_path))
        last_epoch_model_artifact.add_file(str(last_epoch_helper_path))
        run.log_artifact(last_epoch_model_artifact)

        run.name = new_model_id


def get_target_file_paths(directory):
    contents = list(Path(directory).iterdir())
    files = [path for path in contents if path.is_file()]
    return files


def add_em_all(artifact, file_paths):
    for file_path in file_paths:
        artifact.add_file(str(file_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='train_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=YOLO_TRAIN_CONFIG_DIR,
                        help="configuration file directory")
    args_passed = parser.parse_args()

    run_config = get_config(args_passed)
    run_config['host'] = HOST

    load_tune_model(run_config)
