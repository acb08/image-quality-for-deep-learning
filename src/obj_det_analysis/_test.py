# from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# import src.train.train_obj_det
from src.train import rcnn as run
import json
import torch
import src.utils.definitions as definitions
from pathlib import Path

# import src.test.test_obj_det
# import src.test.test_model
from src.utils.detection_functions import listify, yolo_result_to_target_fmt
from ultralytics import YOLO
import torchvision.models as models


def get_model():
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    return models.detection.fasterrcnn_resnet50_fpn(weights=weights)


def evaluate_yolo(model, data_loader, device, status_interval=500):

    cpu_device = torch.device("cpu")

    all_targets = {}
    all_results = {}

    status_interval_crossing = 0

    for i, (images, targets) in enumerate(data_loader):

        results = model(images)
        outputs = [yolo_result_to_target_fmt(result) for result in results]

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

    return all_results, all_targets


def run_test(model, cutoff=None, batch_size=2, output_dir='test_result', output_filename='result.json', log_json=True,
         yolo_mode=True):

    torch.cuda.empty_cache()

    dataset = run.get_dataset(cutoff=cutoff, yolo_fmt=yolo_mode)
    loader = run.get_loader(dataset=dataset,
                            batch_size=batch_size)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    if yolo_mode:
        outputs, targets = evaluate_yolo(model=model,
                                         data_loader=loader,
                                         device=device)
    else:
        outputs, targets = run.evaluate(model=model,
                                            data_loader=loader,
                                            device=device,
                                            yolo_mode=yolo_mode)

    outputs = listify(outputs)
    targets = listify(targets)

    result = {'outputs': outputs, 'targets': targets}

    if log_json:
        output_dir = Path(definitions.ROOT_DIR, output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True, parents=True)

        with open(Path(output_dir, output_filename), 'w') as f:
            json.dump(result, f)

    return outputs, targets


if __name__ == '__main__':

    _cutoff = 16
    _output_dir = f'yolo_rcnn_compare_{_cutoff}-img'
    _yolo = YOLO('yolov8n.pt')

    _yolo_result = run_test(model=_yolo,
                            cutoff=_cutoff,
                            output_dir=_output_dir,
                            output_filename='yolo_result.json',
                            log_json=True,
                            yolo_mode=True)

    _rcnn = get_model()
    _rcnn_result = run_test(model=_rcnn,
                            cutoff=_cutoff,
                            output_dir=_output_dir,
                            output_filename='rcnn_result.json',
                            log_json=True,
                            yolo_mode=False)




