from src.analysis.distortion_performance import load_dataset_and_result, performance_fit_summary_text_dump
from src.utils import detection_functions
import numpy as np
import argparse
from pathlib import Path
from src.utils.functions import get_config
from src.utils import definitions
import wandb
from src.obj_det_analysis.analysis_tools import calculate_aggregate_results
from src.analysis import plot
import time
from src.analysis.analysis_functions import get_sub_dir_and_log_filename, build_3d_field
from src.analysis.fit import fit, evaluate_fit, apply_fit

_REPORT_TIME = False


def _time_string():
    return f'{round(time.time() - _T0, 1)} s'


class ModelDistortionPerformanceResultOD:

    def __init__(self, dataset, result, convert_to_std, result_id, identifier=None, load_local=False,
                 manual_distortion_type_flags=None):

        self.result = result
        self.dataset = dataset
        self.convert_to_std = convert_to_std
        self.result_id = result_id
        self.load_local = load_local
        self.manual_distortion_type_flags = manual_distortion_type_flags
        self.identifier = identifier
        self.dataset_id = result['test_dataset_id']

        self.images = self.dataset['instances']['images']
        self._annotations = self.dataset['instances']['annotations']

        self.distortion_tags = dataset['distortion_tags']
        if 'distortion_type_flags' in dataset.keys():
            self.distortion_type_flags = dataset['distortion_type_flags']
            if manual_distortion_type_flags is not None:
                if set(self.distortion_type_flags) != set(manual_distortion_type_flags):
                    print(f'Warning: distortion type flags ({self.distortion_type_flags}) in dataset differ '
                          f'from manual distortion type flags ({manual_distortion_type_flags})')
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

        self._3d_distortion_perf_props = None

    def __len__(self):
        return len(self.image_ids)

    def __str__(self):
        if self.identifier:
            return str(self.identifier)
        else:
            return self.__repr__()

    def __repr__(self):
        return self.result_id

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
                parsed_outputs = {str(image_id): self.result['outputs'][str(image_id)] for image_id in image_ids}
                parsed_targets = {str(image_id): self.result['targets'][str(image_id)] for image_id in image_ids}

                parsed_mini_results[dist_pt] = {'outputs': parsed_outputs, 'targets': parsed_targets}

            # TODO: figure out why image ids are stored as strings in result['outputs] and result['targets']

            self._parsed_mini_results = parsed_mini_results

        return self._parsed_mini_results

    def get_3d_distortion_perf_props(self, distortion_ids, details=False, make_plots=False, force_recalculate=False):

        if not force_recalculate and self._3d_distortion_perf_props is not None:
            return self._3d_distortion_perf_props

        if distortion_ids != ('res', 'blur', 'noise'):
            raise ValueError('method requires distortion_ids (res, blur, noise)')

        if _REPORT_TIME:
            print(f'getting 3d distortion perf probs, {_time_string()}')

        parsed_mini_results = self.parse_by_dist_pt()

        if _REPORT_TIME:
            print(f'parsed mini results, {_time_string()}')

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

        self._3d_distortion_perf_props = (res_values, blur_values, noise_values, map3d, parameter_array,
                                          performance_array, full_extract)

        return self._3d_distortion_perf_props


def get_obj_det_distortion_perf_result(result_id=None, identifier=None, config=None,
                                       distortion_ids=('res', 'blur', 'noise'), make_dir=True, run=None):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['analysis'], result_id)
    if make_dir and not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    if run is None:
        with wandb.init(project=definitions.WANDB_PID, job_type='analyze_test_result') as run:

            dataset, result, dataset_id = load_dataset_and_result(run=run, result_id=result_id)
            distortion_performance_result = ModelDistortionPerformanceResultOD(
                dataset=dataset,
                result=result,
                convert_to_std=True,
                result_id=result_id,
                identifier=identifier,
                # manual_distortion_type_flags=distortion_ids
            )

    else:
        dataset, result, dataset_id = load_dataset_and_result(run=run, result_id=result_id)
        distortion_performance_result = ModelDistortionPerformanceResultOD(
            dataset=dataset,
            result=result,
            convert_to_std=True,
            result_id=result_id,
            identifier=identifier,
            # manual_distortion_type_flags=distortion_ids
        )

    return distortion_performance_result, output_dir


def flatten_axes_from_cfg(config):

    default = 1, 2, 3

    if 'flatten_axes' in config.keys():

        flatten_axes = config['flatten_axes']

        if flatten_axes == 'default':
            return default

        else:
            return flatten_axes
    else:
        return default


def flatten_axis_combinations_from_cfg(config):

    default = (1, 2), (0, 2), (0, 1)

    if 'flatten_axis_combinations' in config.keys():

        flatten_axis_combinations = config['flatten_axis_combinations']

        if flatten_axis_combinations == 'default':
            return default

        else:
            flatten_axis_combinations = [tuple(combination) for combination in flatten_axis_combinations]
            flatten_axis_combinations = tuple(flatten_axis_combinations)
            return flatten_axis_combinations

    else:
        return default


def fit_keys_from_cfg(config):

    default = ['exp_b0n0', 'pl_b0n0', 'giqe3_b2n2', 'giqe5_b2n2']

    if 'fit_keys' in config.keys():

        fit_keys = config['fit_keys']

        if fit_keys == 'default':
            fit_keys = default
        return fit_keys

    else:
        return default


def view_sorted_performance(performance_array, parameter_array, output_dir=None, low_end=20, high_end=0,
                            show=True):

    indices = np.argsort(performance_array[:, 0])
    performance_sorted = performance_array[indices, 0]

    plot.plot_1d(x=np.arange(len(performance_sorted)),
                 y=performance_sorted,
                 xlabel='index',
                 ylabel='mAP',
                 directory=output_dir,
                 filename='sorted_performance.png',
                 show=show,
                 literal_xlabel=True,
                 literal_ylabel=False)

    res = parameter_array[:, 0][indices]
    blur = parameter_array[:, 1][indices]
    noise = parameter_array[:, 2][indices]

    low_end = min(len(indices), low_end)
    high_end = min(len(indices), high_end)

    if output_dir is not None:
        file = open(Path(output_dir, 'sorted_performance_values.txt'), 'w')
    else:
        file = None

    for i in range(low_end):
        print(f'mAP: {performance_sorted[i]}, ({res[i]}, {blur[i]}, {noise[i]}) (res, blur, noise)', file=file)
    print('\n\n', file=file)
    for j in range(high_end):
        print(f'mAP: {performance_sorted[-j]}, ({res[-j]}, {blur[-j]}, {noise[-j]}) (res, blur, noise)', file=file)

    if file is not None:
        file.close()


def main(config):

    flatten_axes = flatten_axes_from_cfg(run_config)
    flatten_axis_combinations = flatten_axis_combinations_from_cfg(run_config)

    distortion_performance_result, output_dir = get_obj_det_distortion_perf_result(config=run_config)

    res_vals, blur_vals, noise_vals, map3d, parameter_array, perf_array, full_extract = (
        distortion_performance_result.get_3d_distortion_perf_props(distortion_ids=('res', 'blur', 'noise')))

    pass


if __name__ == '__main__':

    _REPORT_TIME = True
    _T0 = time.time()

    ide_config_name = "v8l-fr-10e_fr90-test.yml"  # "v8x_b-scan.yml"  # 'v8n_fr-test.yml'

    if ide_config_name is None:
        config_name = 'distortion_analysis_config.yml'
    else:
        config_name = ide_config_name

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_name, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_analysis_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _flatten_axes = flatten_axes_from_cfg(run_config)
    _flatten_axis_combinations = flatten_axis_combinations_from_cfg(run_config)
    _fit_keys = fit_keys_from_cfg(run_config)

    if 'basic_plots' in run_config.keys():
        basic_plots = run_config['basic_plots']
    else:
        basic_plots = False

    _distortion_performance_result, _output_dir = get_obj_det_distortion_perf_result(config=run_config)

    _res_vals, _blur_vals, _noise_vals, _map3d, _parameter_array, _perf_array, _full_extract = (
        _distortion_performance_result.get_3d_distortion_perf_props(distortion_ids=('res', 'blur', 'noise')))

    view_sorted_performance(_perf_array,
                            _parameter_array,
                            output_dir=_output_dir,
                            low_end=50,
                            high_end=10)

    _perf_dict_3d = {'performance': _map3d}

    if basic_plots and _flatten_axes is not None:

        plot.compare_2d_mean_views(f0=_perf_dict_3d, f1=None,
                                   x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                                   distortion_ids=('res', 'blur', 'noise'),  # flatten_axes=_flatten_axes,
                                   directory=_output_dir,
                                   perf_metric='mAP',
                                   az_el_combinations='all',
                                   show_plots=True)

    if basic_plots and _flatten_axis_combinations is not None:

        plot.plot_1d_from_3d(perf_dict_3d=_perf_dict_3d,
                             x_vals=_res_vals,
                             y_vals=_blur_vals,
                             z_vals=_noise_vals,
                             distortion_ids=('res', 'blur', 'noise'),
                             result_identifier=str(_distortion_performance_result),
                             flatten_axis_combinations=_flatten_axis_combinations,
                             directory=_output_dir,
                             show_plots=True,
                             plot_together=True,
                             ylabel='mAP',
                             legend=False,
                             )

    if _fit_keys is not None:
        _sub_dir, _log_filename = get_sub_dir_and_log_filename(_output_dir, '3d')
        with open(Path(_sub_dir, _log_filename), 'w') as _output_file:

            for _fit_key in _fit_keys:

                _fit_sub_dir, __ = get_sub_dir_and_log_filename(_sub_dir, _fit_key)
                _fit_coefficients = fit(_parameter_array, _perf_array,
                                        distortion_ids=('res', 'blur', 'noise'),
                                        fit_key=_fit_key,
                                        add_bias=False  # only applies to linear bits
                                        )
                _direct_fit_prediction = apply_fit(_fit_coefficients,
                                                   _parameter_array,
                                                   distortion_ids=('res', 'blur', 'noise'),
                                                   fit_key=_fit_key,
                                                   add_bias=False  # only applies to linear fits
                                                   )
                _direct_fit_prediction_3d = build_3d_field(_parameter_array[:, 0],
                                                           _parameter_array[:, 1],
                                                           _parameter_array[:, 2],
                                                           _direct_fit_prediction,
                                                           data_dump=False)
                _fit_correlation = evaluate_fit(_fit_coefficients,
                                                _parameter_array,
                                                _perf_array,
                                                distortion_ids=('res', 'blur', 'noise'),
                                                fit_key=_fit_key,
                                                add_bias=False,  # only applies to linear fits
                                                )
                # plot.compare_2d_mean_views(f0=_map3d, f1=_direct_fit_prediction_3d,
                #                            data_labels=('measured', 'fit'),
                #                            x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                #                            distortion_ids=('res', 'blur', 'noise'),  # flatten_axes=_flatten_axes,
                #                            directory=_fit_sub_dir,
                #                            perf_metric='mAP',
                #                            az_el_combinations='all',
                #                            show_plots=False)

                plot.compare_2d_slice_views(f0=_map3d, f1=_direct_fit_prediction_3d,
                                            data_labels=('measured', 'fit'),
                                            x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                                            distortion_ids=('res', 'blur', 'noise'),  # flatten_axes=_flatten_axes,
                                            directory=_fit_sub_dir,
                                            perf_metric='mAP',
                                            az_el_combinations='default',
                                            show_plots=False
                                            )
                #
                # plot.compare_1d_views(f0=_map3d, f1=_direct_fit_prediction_3d,
                #                       data_labels=('measured', 'fit'),
                #                       x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                #                       flatten_axis_combinations=_flatten_axis_combinations,
                #                       result_id='3d_1d_projection',
                #                       directory=_fit_sub_dir,
                #                       show_plots=True,
                #                       plot_together=True,
                #                       ylabel='mAP')

                print(f'{_fit_key} fit: \n', _fit_coefficients, file=_output_file)
                print(f'{_fit_key} direct fit correlation: ', _fit_correlation, '\n',
                      file=_output_file)


