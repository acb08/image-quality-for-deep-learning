import analysis_tools as tools
import json
from pathlib import Path
import matplotlib.pyplot as plt
import definitions as obj_defs
import src.d00_utils.definitions as proj_defs
from PIL import Image


def load_image(directory, image_id):
    padded_image_id = str(image_id).rjust(12, '0')
    img = Image.open(Path(directory, f'{padded_image_id}.jpg'))
    return img


def load_test_result(dir_name, filename):

    with open(Path(proj_defs.ROOT_DIR, dir_name, filename), 'r') as f:
        data = json.load(f)

    outputs = data['outputs']
    targets = data['targets']

    return outputs, targets


def main(dir_name, filename, dataset_key, num_view_images=5, output_dir=None):

    if output_dir is not None:
        output_dir_abs = Path(proj_defs.ROOT_DIR, output_dir, )
        if not output_dir_abs.is_dir():
            output_dir_abs.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_abs = None

    outputs, targets = load_test_result(dir_name=dir_name, filename=filename)
    image_directory = obj_defs.DATASET_PATHS[dataset_key]['image_dir']
    image_directory = Path(proj_defs.ROOT_DIR, image_directory)
    single_image_results = {}

    for i, (image_id, target) in enumerate(targets.items()):
        predict = outputs[image_id]
        single_image_result, __, __, __ = tools.process_single_image_result(target, predict)
        single_image_results[image_id] = single_image_result

        if i < num_view_images:
            image = load_image(image_directory, image_id)
            tools.visualize_single_image_result(ground_truth=target,
                                                prediction=predict,
                                                image_id=image_id,
                                                output_dir=output_dir_abs,
                                                image=image)

    result, __ = tools.combine_results(single_image_results)

    class_avg_precision_vals = []

    for class_label, data in result.items():
        detections, gt_mapped_scores, gt = data
        precision, recall = tools.raw_pr_curves(detections, gt_mapped_scores, gt)
        precision_smoothed = tools.precision_cleanup(precision)
        ap = tools.get_average_precision(precision_smoothed, recall)
        class_avg_precision_vals.append(ap)

        plt.figure()
        plt.plot(recall, precision)
        plt.plot(recall, precision_smoothed)
        plt.title(f'{class_label}')
        if output_dir_abs is not None:
            plt.savefig(Path(output_dir_abs, f'{class_label}_pr.png'))
        plt.show()


if __name__ == '__main__':

    _dir_name = 'test_result'
    _filename = 'result.json'
    _dataset_key = 'val2017'
    _output_dir = 'analysis_demo'

    main(_dir_name, _filename, _dataset_key, output_dir=_output_dir)

