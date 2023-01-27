from src.d04_analysis.distortion_performance import load_dataset_and_result
from src.d00_utils import detection_functions
import numpy as np
import argparse
from pathlib import Path
from src.d00_utils.functions import get_config
from src.d00_utils import definitions
import wandb
from src.d05_obj_det_analysis.analyze import calculate_aggregate_results
from src.d04_analysis import plot


class ModelDistortionPerformanceResultOD:

    def __init__(self, dataset, result, convert_to_std, identifier=None, load_local=False,
                 manual_distortion_type_flags=None):

        self.result = result
        self.dataset = dataset
        self.convert_to_std = convert_to_std
        self.load_local = load_local
        self.manual_distortion_type_flags = manual_distortion_type_flags
        self.identifier = identifier
        self.dataset_id = result['test_dataset_id']

        self.images = self.dataset['instances']['images']
        self._annotations = self.dataset['instances']['annotations']

        self.distortion_tags = dataset['distortion_tags']
        if 'distortion_type_flags' in dataset.keys():
            self.distortion_type_flags = dataset['distortion_type_flags']
            if set(self.distortion_type_flags) != set(manual_distortion_type_flags):
                print('Warning: distortion type flags in dataset differ from manual distortion type flags')
        else:
            self.distortion_type_flags = manual_distortion_type_flags
        self.convert_to_std = convert_to_std
        self.load_local = load_local
        if self.load_local:
            raise NotImplementedError

        self.image_ids = detection_functions.get_image_ids(self.images)
        self.mapped_boxes_labels = detection_functions.map_boxes_labels(self._annotations, self.image_ids)

        self.distortions = self.build_distortion_vectors()

        if 'res' in self.distortions.keys():
            self.res = self.distortions['res']
        else:
            self.res = np.ones(len(self.image_ids))
            self.distortions['res'] = self.res

        if 'blur' in self.distortions.keys():
            self.blur = self.distortions['blur']
        else:
            self.blur = np.zeros(len(self.image_ids))
            self.distortions['blur'] = self.blur

        if 'noise' in self.distortions.keys():
            self.noise = self.distortions['noise']
        else:
            self.noise = np.zeros(len(self.image_ids))
            self.distortions['noise'] = self.noise

        if self.convert_to_std:
            self.noise = np.sqrt(self.noise)
            self.distortions['noise'] = self.noise

        self.distortion_space = self.get_distortion_space()

        self.image_id_map = self.map_images_to_dist_pts()
        self._parsed_mini_results = None

        self.shape = (len(self.distortion_space[0]), len(self.distortion_space[1]), len(self.distortion_space[2]))

    def __len__(self):
        return len(self.image_ids)

    def build_distortion_vectors(self):
        """
        Pull out distortion info from self._dataset['instances']['images'] and place in numpy vectors
        """
        distortions = {}
        for flag in self.distortion_type_flags:
            distortions[flag] = np.asarray([image[flag] for image in self.images])
        return distortions

    def map_images_to_dist_pts(self):

        if self.res is None or self.blur is None or self.noise is None:
            raise ValueError('')

        res_values, blur_values, noise_values = self.distortion_space

        id_vec = np.asarray(self.image_ids)

        image_id_map = {}

        for i, res_val in enumerate(res_values):
            res_inds = np.where(self.res == res_val)
            for j, blur_val in enumerate(blur_values):
                blur_inds = np.where(self.blur == blur_val)
                for k, noise_val in enumerate(noise_values):
                    noise_inds = np.where(self.noise == noise_val)
                    res_blur_inds = np.intersect1d(res_inds, blur_inds)
                    res_blur_noise_inds = np.intersect1d(res_blur_inds, noise_inds)

                    mapped_image_ids = id_vec[res_blur_noise_inds]

                    image_id_map[(res_val, blur_val, noise_val)] = mapped_image_ids

        return image_id_map

    def get_distortion_space(self):
        return np.unique(self.res), np.unique(self.blur), np.unique(self.noise)

    def get_distortion_matrix(self):
        pass

    def parse_by_dist_pt(self):

        if self._parsed_mini_results is None:

            parsed_mini_results = {}

            for dist_pt, image_ids in self.image_id_map.items():
                parsed_outputs = {k: v for (k, v) in self.result['outputs'].items() if int(k) in image_ids}
                parsed_targets = {k: v for (k, v) in self.result['targets'].items() if int(k) in image_ids}
                parsed_mini_results[dist_pt] = {'outputs': parsed_outputs, 'targets': parsed_targets}

                # TODO: figure out why image ids are stored as strings in result['outputs] and result['targets']
            self._parsed_mini_results = parsed_mini_results

        return self._parsed_mini_results

    def get_3d_distortion_perf_props(self, distortion_ids, details=False, make_plots=False):

        if distortion_ids != ('res', 'blur', 'noise'):
            raise ValueError('method requires distortion_ids (res, blur, noise)')

        parsed_mini_results = self.parse_by_dist_pt()
        res_values, blur_values, noise_values = self.get_distortion_space()

        map3d = np.zeros(self.shape, dtype=np.float32)
        parameter_array = []  # for use in curve fits
        performance_array = []  # for use in svd
        full_extract = {}

        for i, res_val in enumerate(res_values):
            for j, blur_val in enumerate(blur_values):
                for k, noise_val in enumerate(noise_values):

                    dist_pt = (res_val, blur_val, noise_val)
                    mini_result = parsed_mini_results[dist_pt]

                    processed_results = calculate_aggregate_results(outputs=mini_result['outputs'],
                                                                    targets=mini_result['targets'],
                                                                    return_diagnostic_details=details,
                                                                    make_plots=make_plots)

                    class_labels, class_avg_precision_vals, recall, precision, precision_smoothed = processed_results
                    mean_avg_precision = np.mean(class_avg_precision_vals)
                    map3d[i, j, k] = mean_avg_precision
                    parameter_array.append([res_val, blur_val, noise_val])
                    performance_array.append(mean_avg_precision)
                    full_extract[dist_pt] = processed_results

        parameter_array = np.asarray(parameter_array, dtype=np.float32)
        performance_array = np.atleast_2d(np.asarray(performance_array, dtype=np.float32)).T

        return res_values, blur_values, noise_values, map3d, parameter_array, performance_array, full_extract


def get_obj_det_distortion_perf_result(result_id=None, identifier=None, config=None,
                                       distortion_ids=('res', 'blur', 'noise'), make_dir=True):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['analysis'], result_id)
    if make_dir and not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    with wandb.init(project=definitions.WANDB_PID, job_type='analyze_test_result') as run:

        dataset, result, dataset_id = load_dataset_and_result(run=run, result_id=result_id)
        distortion_performance_result = ModelDistortionPerformanceResultOD(dataset=dataset,
                                                                           result=result,
                                                                           convert_to_std=True,
                                                                           identifier=identifier,
                                                                           manual_distortion_type_flags=distortion_ids)

    return distortion_performance_result, output_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_analysis_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_analysis_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _distortion_performance_result, _output_dir = get_obj_det_distortion_perf_result(config=run_config)

    # _outputs = _distortion_performance_result.result['outputs']
    # _targets = _distortion_performance_result.result['targets']

    # _mapped_stuff = _distortion_performance_result.map_images_to_dist_pts()

    # _parsed_results = _distortion_performance_result.parse_by_dist_pt()

    _res_vals, _blur_vals, _noise_vals, _map3d, _parameter_array, _perf_array, _full_extract = (
        _distortion_performance_result.get_3d_distortion_perf_props(distortion_ids=('res', 'blur', 'noise')))

    plot.compare_2d_views(f0=_map3d, f1=_map3d,
                          x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                          distortion_ids=('res', 'blur', 'noise'), flatten_axes=(0, ),
                          directory=_output_dir,
                          perf_metric='mAP')
