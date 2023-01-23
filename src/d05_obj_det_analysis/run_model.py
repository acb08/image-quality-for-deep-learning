from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torch.distributed as dist
import math
import sys
import src.d00_utils.definitions as definitions
from src.d00_utils import functions
from src.d00_utils.classes import COCO


# import numpy as np


def _load_instances_placeholder(dataset_id='val2017'):

    original_dataset = functions.load_original_dataset(dataset_id)
    instances = original_dataset['instances']

    return instances


def get_model():
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)


def get_loader(dataset, batch_size=1):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return loader


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataset(cutoff=None, dataset_key='val2017'):

    path_data = definitions.ORIGINAL_DATASETS[dataset_key]
    image_dir = Path(definitions.ROOT_DIR, path_data['rel_path'])

    instances = _load_instances_placeholder(dataset_id=dataset_key)

    coco = COCO(image_dir, instances, cutoff=cutoff)

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

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        total_images += len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
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

        if i > 0 and i % print_freq == 0:
            print(f'batch {i} train loss:', loss_value)

    mean_loss = running_loss_total / total_images
    print(f'epoch {epoch} mean loss: ', mean_loss)

    return loss_dict, mean_loss


def validate(model, data_loader, device, print_freq=10, scaler=None):

    model.eval()

    running_loss_total = 0
    total_images = 0

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        total_images += len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_list = model(images, targets)
            losses = sum(loss for loss in loss_list)
            running_loss_total += losses

    mean_val_loss = running_loss_total / total_images
    print(f'mean val loss: ', mean_val_loss)

    return mean_val_loss, loss_list


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    all_targets = {}
    all_results = {}

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in out.items()} for out in outputs]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # tgt = {target["image_id"].item(): target for target, output in zip(targets, outputs)}
        tgt = {target["image_id"].item(): target for target in targets}
        all_results.update(res)
        all_targets.update(tgt)

    torch.set_num_threads(n_threads)
    return all_results, all_targets


if __name__ == '__main__':

    if torch.cuda.is_available():
        _device = 'cuda'
        torch.cuda.empty_cache()
    else:
        _device = 'cpu'

    _dataset = get_dataset(cutoff=4)
    _test_dataset = get_dataset(cutoff=20)
    _loader = get_loader(_dataset, batch_size=1)
    _test_loader = get_loader(_test_dataset, batch_size=1)

    _model = get_model()
    _model.to(_device)

    _params = [p for p in _model.parameters() if p.requires_grad]
    _optimizer = torch.optim.SGD(_params, lr=0.005,
                                 momentum=0.9, weight_decay=0.0005)
    _lr_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    _epochs = 1
    _outer_results = []

    for _epoch in range(_epochs):
        # _train_loss_dict, _mean_train_loss = train_one_epoch(model=_model,
        #                                                      optimizer=_optimizer,
        #                                                      data_loader=_loader,
        #                                                      device=_device,
        #                                                      epoch=_epoch,
        #                                                      print_freq=10,
        #                                                      scaler=None)

        _all_outputs, _all_targets = evaluate(model=_model,
                                              data_loader=_test_loader,
                                              device=_device)


    # _model.eval()
    # _images, _targets = next(iter(_loader))
    # _predicts = _model(_images)


