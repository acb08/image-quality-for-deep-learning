import copy
import time
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torch.distributed as dist
import math
import sys
from src.utils.definitions import ROOT_DIR, ORIGINAL_DATASETS, WANDB_PID, STANDARD_DATASET_FILENAME, REL_PATHS, \
    STANDARD_CHECKPOINT_FILENAME, HOST, RCNN_TRAIN_CONFIG_DIR
from src.utils.detection_functions import yolo_result_to_target_fmt, translate_yolo_to_original_label_fmt
from src.utils.functions import load_original_dataset, get_config, construct_artifact_id, load_wandb_data_artifact, \
    load_wandb_model_artifact, id_from_tags, wandb_to_detection_dataset
from src.utils.classes import COCO
import argparse
import wandb
from src.train.train import log_checkpoint, save_best_loss_model


_T0 = time.time()


def _load_instances_placeholder(dataset_id='val2017'):

    original_dataset = load_original_dataset(dataset_id)
    instances = original_dataset['instances']

    return instances


def get_loader(dataset, batch_size=1, num_workers=0):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )
    return loader


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataset(cutoff=None, dataset_key='val2017', yolo_fmt=False):

    path_data = ORIGINAL_DATASETS[dataset_key]
    image_dir = Path(ROOT_DIR, path_data['rel_path'])

    instances = _load_instances_placeholder(dataset_id=dataset_key)

    coco = COCO(image_dir, instances, cutoff=cutoff, yolo_fmt=yolo_fmt)

    return coco


# from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None):

    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    running_loss_total = 0
    total_images = 0

    num_batches = len(data_loader)

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        total_images += len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_loss_total += losses

            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if i > 0 and total_images % print_freq == 0:
                print(f'batch {i + 1} / {num_batches} train loss ({total_images} images):', loss_value)

        except AssertionError:
            print(f'AssertionError, batch {i}')

    mean_loss = float(running_loss_total) / total_images
    print(f'epoch {epoch} mean loss: ', mean_loss)

    return loss_dict, mean_loss


def validate(model, data_loader, device, print_freq=10, scaler=None):

    with torch.no_grad():

        model.train()  # if model in eval mode, boxes returned rather than loss values

        running_loss_total = 0
        total_images = 0

        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            total_images += len(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss.item() for loss in loss_dict.values())
                running_loss_total += losses

        mean_val_loss = running_loss_total / total_images
        print(f'mean val loss: ', mean_val_loss)

    return loss_dict, mean_val_loss


@torch.inference_mode()
def evaluate(model, data_loader, device, status_interval=500, yolo_mode=False):

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    if not yolo_mode:
        model.eval()

    all_targets = {}
    all_results = {}

    status_interval_crossing = 0

    for i, (images, targets) in enumerate(data_loader):

        if not yolo_mode:
            images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if yolo_mode:

            results = model(images, verbose=False)

            outputs = [yolo_result_to_target_fmt(result) for result in results]
            outputs = [{k: v.to(cpu_device) for k, v in out.items()} for out in outputs]
            outputs = [translate_yolo_to_original_label_fmt(out) for out in outputs]

            targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        else:
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in out.items()} for out in outputs]
            targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        tgt = {target["image_id"].item(): target for target in targets}
        all_results.update(res)
        all_targets.update(tgt)

        total_images = (i + 1) * len(images)
        total_status_intervals = divmod(total_images, status_interval)[0]
        if total_status_intervals > status_interval_crossing:
            status_interval_crossing = total_status_intervals
            print(f'{total_images} images complete')

    torch.set_num_threads(n_threads)
    return all_results, all_targets


def load_tune_model(config):

    with wandb.init(project=WANDB_PID, job_type='train_model', config=config) as run:

        config = wandb.config  # allows wandb parameter sweeps (not currently implemented)

        dataset_artifact_id, __ = construct_artifact_id(config['train_dataset_id'],
                                                        artifact_alias=config['train_dataset_artifact_alias'])
        starting_model_artifact_id, __ = construct_artifact_id(config['starting_model_id'],
                                                               artifact_alias=config['starting_model_artifact_alias'])

        model, arch, __ = load_wandb_model_artifact(run, starting_model_artifact_id, return_configs=True)
        config['arch'] = arch

        __, dataset = load_wandb_data_artifact(run, dataset_artifact_id, STANDARD_DATASET_FILENAME)
        detection_dataset = wandb_to_detection_dataset(dataset, yolo_fmt=False)

        val_sibling_dataset_id = dataset['val_sibling_dataset_id']
        val_sibling_dataset_id, __ = construct_artifact_id(val_sibling_dataset_id)
        __, val_dataset = load_wandb_data_artifact(run, val_sibling_dataset_id, STANDARD_DATASET_FILENAME)
        val_detection_dataset = wandb_to_detection_dataset(val_dataset, yolo_fmt=False)

        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        artifact_type = config['artifact_type']
        status_interval = config['status_interval']
        description = config['description']

        optimizer_type = config['optimizer_type']
        lr = config['lr']
        momentum = config['momentum']
        weight_decay = config['weight_decay']

        model_id_tags = [arch]
        if 'name_string' in config.keys() and config['name_string'] is not None:
            model_id_tags.append(config['name_string'])
            new_model_id = id_from_tags(artifact_type, model_id_tags)

        else:
            new_model_id = id_from_tags(artifact_type, model_id_tags)

        new_model_rel_dir = Path(REL_PATHS[artifact_type], new_model_id)
        output_dir = Path(ROOT_DIR, new_model_rel_dir)
        output_dir.mkdir(parents=True)

        if 'device' in config.keys():
            device = config['device']
            if type(device) == str:
                if device.isdigit():
                    device = int(device)
            if type(device) == int:
                device = f'cuda:{device}'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device: ', device)

        loader = get_loader(detection_dataset, num_workers=num_workers, batch_size=batch_size)
        val_loader = get_loader(val_detection_dataset, num_workers=num_workers, batch_size=batch_size)

        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        # optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = getattr(torch.optim, optimizer_type)(params, lr=lr,
                                                         momentum=momentum, weight_decay=weight_decay)

        new_model_checkpoint_file_config = {
            'model_rel_dir': str(new_model_rel_dir),
            'model_filename': STANDARD_CHECKPOINT_FILENAME
        }
        config['model_file_config'] = new_model_checkpoint_file_config

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):

            train_loss_dict, mean_train_loss = train_one_epoch(model=model,
                                                               optimizer=optimizer,
                                                               data_loader=loader,
                                                               device=device,
                                                               epoch=epoch,
                                                               print_freq=status_interval,
                                                               scaler=None)
            train_losses.append(mean_train_loss)

            loss_dict, mean_val_loss = validate(model=model,
                                                data_loader=val_loader,
                                                device=device,
                                                scaler=None)
            val_losses.append(mean_val_loss)

            artifact_metadata = dict(config)
            artifact_metadata.update({
                'epoch': epoch,
            })

            new_model_artifact = wandb.Artifact(
                new_model_id,
                type=artifact_type,
                metadata=artifact_metadata,
                description=description,
            )

            _time = round(time.time() - _T0, 1)
            epoch_stats = {
                'train_loss': mean_train_loss,
                'val_loss': mean_val_loss,
                'epoch': epoch,
                'time': _time
            }
            wandb.log(epoch_stats)
            log_checkpoint(model, artifact_metadata, new_model_artifact, run)

            if val_losses[-1] <= min(val_losses):

                best_loss_model_metadata = copy.deepcopy(dict(artifact_metadata))
                best_loss_model_path, best_loss_helper_path = save_best_loss_model(model,
                                                                                   best_loss_model_metadata,
                                                                                   val_losses,
                                                                                   epoch)

            print(f'Epoch {epoch} losses: {mean_train_loss} (train), {mean_train_loss} (val), {_time} s run time')

        best_loss_model_artifact = wandb.Artifact(
            f'{new_model_id}_best_loss',
            type=artifact_type,
            metadata=best_loss_model_metadata,
            description=description
        )
        best_loss_model_artifact.add_file(str(best_loss_model_path))
        best_loss_model_artifact.add_file(str(best_loss_helper_path))
        run.log_artifact(best_loss_model_artifact)
        run.name = new_model_id

    print('done')


if __name__ == '__main__':

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='train_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=RCNN_TRAIN_CONFIG_DIR,
                        help="configuration file directory")
    args_passed = parser.parse_args()

    run_config = get_config(args_passed)
    run_config['host'] = HOST

    load_tune_model(run_config)

    # if torch.cuda.is_available():
    #     _device = 'cuda'
    #     torch.cuda.empty_cache()
    # else:
    #     _device = 'cpu'
    #
    # _dataset = get_dataset(cutoff=4)
    # _test_dataset = get_dataset(cutoff=20)
    # _loader = get_loader(_dataset, batch_size=1)
    # _test_loader = get_loader(_test_dataset, batch_size=1)

    # _model = get_model()
    # _model.to(_device)

    # _params = [p for p in _model.parameters() if p.requires_grad]
    # _optimizer = torch.optim.SGD(_params, lr=0.005,
    #                              momentum=0.9, weight_decay=0.0005)
    # _lr_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer,
    #                                                 step_size=3,
    #                                                 gamma=0.1)
    #
    # _epochs = 1
    # _outer_results = []
    #
    # for _epoch in range(_epochs):
    #     # _train_loss_dict, _mean_train_loss = train_one_epoch(model=_model,
    #     #                                                      optimizer=_optimizer,
    #     #                                                      data_loader=_loader,
    #     #                                                      device=_device,
    #     #                                                      epoch=_epoch,
    #     #                                                      print_freq=10,
    #     #                                                      scaler=None)
    #
    #     _all_outputs, _all_targets = evaluate(model=_model,
    #                                           data_loader=_test_loader,
    #                                           device=_device)


    # _model.eval()
    # _images, _targets = next(iter(_loader))
    # _predicts = _model(_images)


