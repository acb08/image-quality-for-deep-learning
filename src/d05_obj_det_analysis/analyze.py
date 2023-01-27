import analysis_tools as tools
import json
from pathlib import Path
import matplotlib.pyplot as plt
import src.d00_utils.definitions as definitions
from PIL import Image
import numpy as np


def load_image(directory, image_id, extension='png'):
    padded_image_id = str(image_id).rjust(12, '0')
    img = Image.open(Path(directory, f'{padded_image_id}.{extension}'))
    return img


def load_test_result(dir_name, filename):

    with open(Path(definitions.ROOT_DIR, dir_name, filename), 'r') as f:
        data = json.load(f)

    outputs = data['outputs']
    targets = data['targets']

    return outputs, targets


def calculate_aggregate_results(outputs, targets,
                                return_diagnostic_details=False, make_plots=True, output_dir_abs=None):

    single_image_results = {}

    for i, (image_id, target) in enumerate(targets.items()):
        predict = outputs[image_id]
        single_image_result, __, __, __ = tools.process_single_image_result(target, predict)
        single_image_results[image_id] = single_image_result

    result, __ = tools.combine_results(single_image_results)

    class_avg_precision_vals = []
    class_labels = []

    if return_diagnostic_details:
        recall_curves = []
        precision_curves = []
        precision_curves_smoothed = []

    else:
        recall_curves = None
        precision_curves = None
        precision_curves_smoothed = None

    for class_label, data in result.items():

        detections, gt_mapped_scores, gt = data
        precision, recall = tools.raw_pr_curves(detections, gt_mapped_scores, gt)
        precision_smoothed = tools.precision_cleanup(precision)
        ap = tools.get_average_precision(precision_smoothed, recall)

        class_avg_precision_vals.append(ap)
        class_labels.append(class_label)

        if return_diagnostic_details:
            recall_curves.append(recall)
            precision_curves.append(precision)
            precision_curves_smoothed.append(precision_smoothed)

        if make_plots:
            plt.figure()
            plt.plot(recall, precision)
            plt.plot(recall, precision_smoothed)
            plt.title(f'{class_label}')
            if output_dir_abs is not None:
                plt.savefig(Path(output_dir_abs, f'{class_label}_pr.png'))
            plt.show()

    return class_labels, class_avg_precision_vals, recall_curves, precision_curves, precision_curves_smoothed


def main(dir_name, filename, num_view_images=5, output_dir=None, image_directory=None):

    if output_dir is not None:
        output_dir_abs = Path(definitions.ROOT_DIR, output_dir, )
        if not output_dir_abs.is_dir():
            output_dir_abs.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_abs = None

    outputs, targets = load_test_result(dir_name=dir_name, filename=filename)
    image_directory = Path(definitions.ROOT_DIR, image_directory)

    for i in range(num_view_images):
        image_id, target = list(targets.items())[i]
        predict = outputs[image_id]
        image = load_image(image_directory, image_id)
        tools.visualize_single_image_result(ground_truth=target,
                                            prediction=predict,
                                            image_id=image_id,
                                            output_dir=output_dir_abs,
                                            image=image)

    class_avg_precision_vals, __, __, __, __ = calculate_aggregate_results(outputs=outputs,
                                                                           targets=targets,
                                                                           return_diagnostic_details=False,
                                                                           make_plots=True,
                                                                           output_dir_abs=output_dir_abs)

    mean_avg_precision = np.mean(class_avg_precision_vals)
    print('mAP: ', mean_avg_precision)

    # single_image_results = {}
    #
    # for i, (image_id, target) in enumerate(targets.items()):
    #     predict = outputs[image_id]
    #     single_image_result, __, __, __ = tools.process_single_image_result(target, predict)
    #     single_image_results[image_id] = single_image_result
    #
    #     # if i < num_view_images:
    #     #     image = load_image(image_directory, image_id)
    #     #     tools.visualize_single_image_result(ground_truth=target,
    #     #                                         prediction=predict,
    #     #                                         image_id=image_id,
    #     #                                         output_dir=output_dir_abs,
    #     #                                         image=image)
    #
    # result, __ = tools.combine_results(single_image_results)
    #
    # class_avg_precision_vals = []
    #
    # for class_label, data in result.items():
    #     detections, gt_mapped_scores, gt = data
    #     precision, recall = tools.raw_pr_curves(detections, gt_mapped_scores, gt)
    #     precision_smoothed = tools.precision_cleanup(precision)
    #     ap = tools.get_average_precision(precision_smoothed, recall)
    #     class_avg_precision_vals.append(ap)
    #
    #     plt.figure()
    #     plt.plot(recall, precision)
    #     plt.plot(recall, precision_smoothed)
    #     plt.title(f'{class_label}')
    #     if output_dir_abs is not None:
    #         plt.savefig(Path(output_dir_abs, f'{class_label}_pr.png'))
    #     plt.show()


if __name__ == '__main__':

    _dir_name = 'test_results/0008rlt-fasterrcnn_resnet50_fpn-0041tst-coco_rbn_checkout-res-blur-noise'
    _filename = 'test_result.json'
    _dataset_key = '0041tst-coco_rbn_checkout'
    _output_dir = 'quick_look_rbn_initial_checkout'
    _image_directory = Path(definitions.ROOT_DIR, 'datasets/test', _dataset_key)

    main(_dir_name, _filename, num_view_images=64, output_dir=_output_dir, image_directory=_image_directory)

