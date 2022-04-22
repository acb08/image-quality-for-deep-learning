from src.d00_utils.definitions import ROOT_DIR
# from src.d04_analysis.image_property_extractor_proto import property_extractor
from pathlib import Path
import json
import numpy as np


def property_extractor(dictionary, target_key, iter_keys=None, traverse_keys=None):

    values = []

    if not traverse_keys:
        traverse_keys = []
    traverse_keys.append(target_key)

    if not iter_keys:
        iter_keys = list(dictionary.keys)

    for iter_key in iter_keys:

        inner_dict = dictionary[iter_key]
        for sub_key in traverse_keys:
            inner_dict = inner_dict[sub_key]
        values.append(inner_dict)

    return values


# pan_c_dir = r'/cis/phd/acb6595/places/results/manual/crop_224/resnet18-_pan_c_best_loss'
pan_c_dir = r'results/manual/crop_224/resnet18-_pan_c_best_loss'

# noise corrected endpoint
# pan_c_r6_b5_nf5_dir = r'/cis/phd/acb6595/places/results/manual/crop_224/resnet18-_pan_c_r6_b5_nf5_best_loss'
pan_c_r6_b5_nf5_dir = r'results/manual/crop_224/resnet18-_pan_c_r6_b5_nf5_best_loss'

# noise corrected full range
# pan_c_r10_b10_nf11_dir = r'/cis/phd/acb6595/places/results/manual/crop_224/resnet18-_pan_c_r10_b10_nf11_best_loss'
pan_c_r10_b10_nf11_dir = r'results/manual/crop_224/resnet18-_pan_c_r10_b10_nf11_best_loss'

# noise corrected mid-band
pan_c_r9_b9_nf10_dir = r'/cis/phd/acb6595/places/results/manual/crop_224/resnet18-_pan_c_r9_b9_nf10_best_loss'
pan_c_r9_b9_nf10_dir = r'results/manual/crop_224/resnet18-_pan_c_r9_b9_nf10_best_loss'

# noise corrected midpoint
# pan_c_r4_b4_nf4_dir = r'/cis/phd/acb6595/places/results/manual/crop_224/resnet18-_pan_c_r4_b4_nf4_best_loss'
pan_c_r4_b4_nf4_dir = r'results/manual/crop_224/resnet18-_pan_c_r4_b4_nf4_best_loss'

pre_trained = {
    'identifier': 'pre_trained',
    'rel_dir': pan_c_dir
}

endpoint = {

}

results_dict = {
    'pre-trained': {
        'identifier': 'pre-trained',
        'rel_dir': pan_c_dir
    },
    'midpoint': {
        'identifier': 'midpoint',
        'rel_dir': pan_c_r4_b4_nf4_dir
    },
    'mid-band': {
        'identifier': 'mid-band',
        'rel_dir': pan_c_r9_b9_nf10_dir
    },
    'endpoint': {
        'identifier': 'endpoint',
        'rel_dir': pan_c_r6_b5_nf5_dir
    },
    'full-range': {
        'identifier': 'full-range',
        'rel_dir': pan_c_r10_b10_nf11_dir
    },
}

MANUAL_RESULTS_FILENAME = r'img_dist_224_pan_msc1_rc3_bc3_nf3_test_results.json'
DISTORTION_PROPERTIES_FILENAME = r'metadata_pan_msc1_rc3_bc3_nf3.json'


def load_result(result_tag):

    result_info = results_dict[result_tag]
    result_path = Path(ROOT_DIR, result_info['rel_dir'], MANUAL_RESULTS_FILENAME)
    with open(result_path, 'r') as file:
        performance_result = json.load(file)

    distortion_data_path = Path(ROOT_DIR, DISTORTION_PROPERTIES_FILENAME)
    with open(distortion_data_path, 'r') as file:
        distortion_data = json.load(file)

    image_names = []
    names_labels = distortion_data['names_labels']
    for name, label in names_labels:
        image_names.append(name)

    distortion_params = property_extractor(distortion_data['image_info'], 'distortion_params', iter_keys=image_names)

    blur_stds = []
    noise_means = []
    res_fractions = []

    for param_set in distortion_params:
        res_fractions.append(param_set[1][1])
        blur_stds.append(param_set[2][1])
        noise_means.append(param_set[3][1])

    blur_stds = np.asarray(blur_stds)
    noise_means = np.asarray(noise_means)
    # if noise_specification == 'variance':
    # noise_means = np.sqrt(noise_means)
    res_fractions = np.asarray(res_fractions)

    distortion_matrix = np.zeros((len(distortion_params), 3))
    distortion_matrix[:, 0] = res_fractions
    distortion_matrix[:, 1] = blur_stds
    distortion_matrix[:, 2] = noise_means

    return distortion_matrix, performance_result, result_info['identifier']


if __name__ == '__main__':

    perf_result, distortion_info, result_id = load_result('pre-trained')
    # print('hi')
