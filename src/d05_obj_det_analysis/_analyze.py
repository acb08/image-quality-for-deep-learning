import src.d05_obj_det_analysis.analysis_tools as tools
from src.d05_obj_det_analysis.analysis_tools import calculate_aggregate_results
import json
from pathlib import Path
import src.d00_utils.definitions as definitions
from PIL import Image
import numpy as np


def load_image(directory, image_id, extension='jpg'):
    padded_image_id = str(image_id).rjust(12, '0')
    img = Image.open(Path(directory, f'{padded_image_id}.{extension}'))
    return img


def load_test_result(dir_name, filename):

    with open(Path(definitions.ROOT_DIR, dir_name, filename), 'r') as f:
        data = json.load(f)

    outputs = data['outputs']
    targets = data['targets']

    return outputs, targets


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

    class_labels, class_avg_precision_vals, __, __, __ = calculate_aggregate_results(outputs=outputs,
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
    #
    # _dir_name = '/home/acb6595/coco/test_results/0023rlt-fasterrcnn_resnet50_fpn-val2017'
    # _filename = 'test_result.json'
    # _dataset_key = 'val2017'
    # _output_dir = r'yolo_debug-rcnn-compare/val2017'
    # _image_directory = Path(definitions.ROOT_DIR, 'datasets/test', _dataset_key)
    #
    # # main(_dir_name, _filename, num_view_images=5, output_dir=_output_dir, image_directory=_image_directory)
    # _result = load_test_result(dir_name=_dir_name, filename=_filename)

    from src.d00_utils.definitions import YOLO_TO_ORIGINAL_PAPER_KEY_MAPPING
    print(YOLO_TO_ORIGINAL_PAPER_KEY_MAPPING)


