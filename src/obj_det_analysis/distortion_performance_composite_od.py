import copy

from src.obj_det_analysis.load_multiple_od_results import get_multiple_od_distortion_performance_results
from src.utils.functions import get_config
import numpy as np
import argparse
from pathlib import Path
from src.utils.definitions import DISTORTION_RANGE_90
import matplotlib.pyplot as plt
from src.analysis.distortion_performance_composite import get_composite_performance_result_output_directory, \
    get_composite_result_uid
from src.analysis.analysis_functions import get_sub_dir_and_log_filename, build_3d_field
from src.analysis.fit import fit, evaluate_fit, apply_fit
from src.analysis.aic import akaike_info_criterion
from src.analysis import plot
from src.obj_det_analysis.distortion_performance_od import flatten_axis_combinations_from_cfg, fit_keys_from_cfg, \
    flatten_axes_from_cfg
from src.utils.shared_methods import _get_processed_instance_props_path, _archive_processed_props, \
    _check_extract_processed_props
from src.utils.vc import get_od_composite_hash_mash
from hashlib import blake2b


def oct_vec_to_int(oct_vec):

    assert np.shape(oct_vec) == (3,)
    assert set(np.unique(oct_vec)).issubset({0, 1})

    bin_val = f'0b{int(oct_vec[0])}{int(oct_vec[1])}{int(oct_vec[2])}'

    return int(bin_val, 2)


def label_octant_array(oct_array):

    m, n = np.shape(oct_array)
    assert n == 3

    oct_labels = -1 * np.ones(m)

    for i in range(m):
        oct_vec = oct_array[i]
        oct_num = oct_vec_to_int(oct_vec)
        oct_labels[i] = oct_num

    assert -1 not in oct_labels

    return oct_labels


def load_multiple_3d_perf_results(result_id_pairs, distortion_ids):

    distortion_performance_results = get_multiple_od_distortion_performance_results(result_id_pairs,
                                                                                    output_type='list')

    res_vals, blur_vals, noise_vals = None, None, None
    parameter_array = None

    perf_results_3d = {}
    perf_arrays = {}
    model_artifact_ids = set()

    length = None

    for i, distortion_performance_result_od in enumerate(distortion_performance_results):

        model_artifact_id = distortion_performance_result_od.model_artifact_id

        _res_vals, _blur_vals, _noise_vals, _map3d, _parameter_array, _perf_array, _full_extract = \
            distortion_performance_result_od.get_3d_distortion_perf_props(distortion_ids=distortion_ids)

        if i == 0:
            res_vals, blur_vals, noise_vals = _res_vals, _blur_vals, _noise_vals
            length = len(_perf_array)
        else:
            assert np.array_equal(res_vals, _res_vals)
            assert np.array_equal(blur_vals, _blur_vals)
            assert np.array_equal(noise_vals, _noise_vals)
            assert length == len(_perf_array)

        if i == 0:
            parameter_array = _parameter_array
        else:
            assert np.array_equal(parameter_array, _parameter_array)

        assert model_artifact_id not in model_artifact_ids  # redundant assertions, being careful / maintaining symmetry
        assert model_artifact_id not in perf_results_3d.keys()
        assert model_artifact_id not in perf_arrays.keys()
        model_artifact_ids.add(model_artifact_id)
        perf_results_3d[model_artifact_id] = _map3d
        perf_arrays[model_artifact_id] = _perf_array

    return res_vals, blur_vals, noise_vals, perf_results_3d, parameter_array, perf_arrays, model_artifact_ids, length


class CompositePerformanceResultOD:
    """
    Intended to act like CompositePerformanceResult for object detections, simplified to used only 3d distortion
    performance arrays
    """

    def __init__(self, performance_prediction_result_id_pairs, performance_eval_result_id_pairs, identifier,
                 distortion_ids=('res', 'blur', 'noise'), res_boundary_weights=None, ignore_vc_hashes=False):

        if res_boundary_weights is None:
            res_boundary_weights = (1, 1)

        self.performance_prediction_result_id_pairs = performance_prediction_result_id_pairs
        self.performance_eval_result_id_pairs = performance_eval_result_id_pairs
        self.identifier = identifier
        self.distortion_ids = distortion_ids

        self.performance_predict_result_ids = [pair[0] for pair in self.performance_prediction_result_id_pairs]
        self.performance_eval_result_ids = [pair[0] for pair in self.performance_eval_result_id_pairs]

        perf_predict_uid_hash_data = copy.deepcopy(self.performance_predict_result_ids)
        perf_predict_uid_hash_data.append(str(res_boundary_weights))
        self.uid = get_composite_result_uid(
            performance_prediction_result_ids=perf_predict_uid_hash_data,
            eval_result_ids=self.performance_eval_result_ids
        )

        predict_hash, eval_hash = self._get_instance_hashes()
        self.instance_hashes = {'predict': predict_hash, 'eval': eval_hash}
        self.ignore_vc_hashes = ignore_vc_hashes

        self.result_id = self.__repr__()
        self._res_lower_weight, self._res_upper_weight = res_boundary_weights
        self.vc_hash_mash = get_od_composite_hash_mash()

        self._3d_distortion_perf_props = {}

        predict_extract = self.check_extract_processed_props(predict_eval_flag='predict')
        eval_extract = self.check_extract_processed_props(predict_eval_flag='eval')
        if predict_extract and eval_extract:
            self._3d_distortion_perf_props = {'predict': predict_extract, 'eval': eval_extract}
            print('loaded cached composite performance props')
        else:
            _predict_data = load_multiple_3d_perf_results(self.performance_prediction_result_id_pairs,
                                                          self.distortion_ids)
            self._perf_3d_predict = _predict_data[3]  # perf_results_3d, parameter_array, perf_arrays
            self._distortion_array_predict = _predict_data[4]
            self._performance_arrays_predict = _predict_data[5]
            self._model_artifact_id_set = _predict_data[6]
            self._length = _predict_data[7]
            del _predict_data

            _eval_data = load_multiple_3d_perf_results(self.performance_eval_result_id_pairs,
                                                       self.distortion_ids)

            self.res_vals = _eval_data[0]
            self.blur_vals = _eval_data[1]
            self.noise_vals = _eval_data[2]
            self._perf_3d_eval = _eval_data[3]
            self._distortion_array = _eval_data[4]
            self._performance_arrays = _eval_data[5]
            _eval_model_artifact_ids = _eval_data[6]
            _eval_length = _eval_data[7]
            del _eval_data

            assert self._performance_arrays.keys() == self._performance_arrays_predict.keys()
            assert _eval_model_artifact_ids == self._model_artifact_id_set
            assert set(self._performance_arrays.keys()) == self._model_artifact_id_set
            assert self._length == _eval_length

            self._model_artifact_ids = tuple(self._model_artifact_id_set)
            del self._model_artifact_id_set  # ensure constant model id order by using self.model_artifact_ids tuple

            self.res_boundary, self.blur_boundary, self.noise_boundary = self._get_octant_boundaries()
            self._predict_oct_vectors, self._eval_oct_vectors = self._assign_octant_vectors()

            self._predict_oct_labels = label_octant_array(self._predict_oct_vectors)
            self._eval_oct_labels = label_octant_array(self._eval_oct_vectors)

            self._oct_nums = np.unique(self._eval_oct_labels)
            assert np.array_equal(self._oct_nums, np.unique(self._predict_oct_labels))
            self._oct_nums = tuple(self._oct_nums)  # make immutable just to be safe

            (self._predict_model_oct_performance, self._eval_model_oct_performance,
             self._predict_model_oct_performance_dict,
             self._eval_model_oct_performance_dict) = self._get_performance_by_octant()
            self._octant_model_map = self._assign_best_models_to_octants()

            self._predict_composite_performance, self._eval_composite_performance = self._composite_performance()

    def __len__(self):
        return self._length

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return f'{self.__str__()}-{self.uid}'

    def _get_instance_hashes(self):

        predict_hash_string = f'{self.uid}-predict'
        eval_hash_string = f'{self.uid}-eval'

        predict_hash = blake2b(predict_hash_string.encode('utf8')).hexdigest()
        eval_hash = blake2b(eval_hash_string.encode('utf8')).hexdigest()

        return predict_hash, eval_hash

    def _get_octant_boundaries(self):

        distortion_range = DISTORTION_RANGE_90['coco']
        res_vals = distortion_range['res']
        blur_vals = distortion_range['blur']
        noise_vals = distortion_range['noise']

        assert np.array_equal(res_vals, self.res_vals)
        assert np.array_equal(blur_vals, self.blur_vals)
        assert np.array_equal(noise_vals, self.noise_vals)

        res_min, res_max = np.min(res_vals), np.max(res_vals)
        blur_min, blur_max = np.min(blur_vals), np.max(blur_vals)
        noise_min, noise_max = np.min(noise_vals), np.max(noise_vals)

        res_weighting_total = self._res_lower_weight + self._res_upper_weight
        res_boundary = (self._res_lower_weight * res_min + self._res_upper_weight * res_max) / res_weighting_total
        blur_boundary = (blur_min + blur_max) / 2
        noise_boundary = (noise_min + noise_max) / 2

        return res_boundary, blur_boundary, noise_boundary

    def get_processed_instance_props_path(self, predict_eval_flag):
        return _get_processed_instance_props_path(self, predict_eval_flag=predict_eval_flag)

    def archive_processed_props(self, predict_eval_flag: object) -> object:

        data = self.get_3d_distortion_perf_props(predict_eval_flag=predict_eval_flag)
        res_vals, blur_vals, noise_vals, map_3d, distortion_array, performance_array, __ = data

        return _archive_processed_props(self, res_values=res_vals, blur_values=blur_vals, noise_values=noise_vals,
                                        perf_3d=map_3d, distortion_array=distortion_array, perf_array=performance_array,
                                        predict_eval_flag=predict_eval_flag, vc_hash_mash=self.vc_hash_mash)

    def check_extract_processed_props(self, predict_eval_flag):
        return _check_extract_processed_props(self, predict_eval_flag=predict_eval_flag)

    def _assign_octant_vectors(self):

        predict_oct_ids = -1 * np.ones_like(self._distortion_array_predict)

        res_predict = self._distortion_array_predict[:, 0]
        blur_predict = self._distortion_array_predict[:, 1]
        noise_predict = self._distortion_array_predict[:, 2]

        predict_res_labels = -1 * np.ones_like(res_predict)
        predict_res_labels[np.where(res_predict >= self.res_boundary)] = 0
        predict_res_labels[np.where(res_predict < self.res_boundary)] = 1
        assert -1 not in predict_res_labels

        predict_blur_labels = -1 * np.ones_like(blur_predict)
        predict_blur_labels[np.where(blur_predict <= self.blur_boundary)] = 0
        predict_blur_labels[np.where(blur_predict > self.blur_boundary)] = 1
        assert -1 not in predict_blur_labels

        predict_noise_labels = -1 * np.ones_like(noise_predict)
        predict_noise_labels[np.where(noise_predict <= self.noise_boundary)] = 0
        predict_noise_labels[np.where(noise_predict > self.noise_boundary)] = 1
        assert -1 not in predict_noise_labels

        predict_oct_ids[:, 0] = predict_res_labels
        predict_oct_ids[:, 1] = predict_blur_labels
        predict_oct_ids[:, 2] = predict_noise_labels

        eval_oct_ids = -1 * np.ones_like(self._distortion_array)
        res = self._distortion_array[:, 0]
        blur = self._distortion_array[:, 1]
        noise = self._distortion_array[:, 2]

        res_labels = -1 * np.ones_like(res)
        res_labels[np.where(res >= self.res_boundary)] = 0
        res_labels[np.where(res < self.res_boundary)] = 1
        assert -1 not in res_labels

        blur_labels = -1 * np.ones_like(blur)
        blur_labels[np.where(blur <= self.blur_boundary)] = 0
        blur_labels[np.where(blur > self.blur_boundary)] = 1
        assert -1 not in blur_labels

        noise_labels = -1 * np.ones_like(noise)
        noise_labels[np.where(noise <= self.noise_boundary)] = 0
        noise_labels[np.where(noise > self.noise_boundary)] = 1
        assert -1 not in noise_labels

        eval_oct_ids[:, 0] = res_labels
        eval_oct_ids[:, 1] = blur_labels
        eval_oct_ids[:, 2] = noise_labels
        assert -1 not in eval_oct_ids

        return predict_oct_ids, eval_oct_ids

    def _get_performance_by_octant(self, verbose=False):

        predict_oct_performances = {}
        eval_oct_performances = {}

        num_models = len(self._model_artifact_ids)
        num_octants = len(self._oct_nums)
        assert num_octants == 8

        predict_oct_performance_array = -1 * np.ones((num_models, num_octants))
        eval_oct_performance_array = -1 * np.ones((num_models, num_octants))

        for model_index, model_id in enumerate(self._model_artifact_ids):

            predict_performance_array = np.ravel(self._performance_arrays_predict[model_id])
            eval_performance_array = np.ravel(self._performance_arrays[model_id])
            predict_extraction_total = 0
            eval_extraction_total = 0

            for octant_index, oct_num in enumerate(self._oct_nums):

                assert octant_index == oct_num

                predict_extractor = self._make_extractor(oct_num=oct_num, predict_eval_flag='predict')

                num_predict_points = np.sum(predict_extractor)
                predict_extraction_total += num_predict_points

                predict_oct_avg_performance = np.sum(predict_extractor * predict_performance_array) / num_predict_points
                self._update_oct_performance_array(
                    oct_performance_array=predict_oct_performance_array,
                    val=predict_oct_avg_performance,
                    model_index=model_index,
                    octant_index=octant_index
                )

                if model_id not in predict_oct_performances.keys():
                    predict_oct_performances[model_id] = {oct_num: predict_oct_avg_performance}
                else:
                    predict_oct_performances[model_id][oct_num] = predict_oct_avg_performance

                eval_extractor = self._make_extractor(oct_num=oct_num, predict_eval_flag='eval')
                num_eval_points = np.sum(eval_extractor)
                eval_extraction_total += num_eval_points

                eval_oct_avg_performance = np.sum(eval_extractor * eval_performance_array) / num_eval_points
                self._update_oct_performance_array(
                    oct_performance_array=eval_oct_performance_array,
                    val=eval_oct_avg_performance,
                    model_index=model_index,
                    octant_index=octant_index
                )

                if model_id not in eval_oct_performances.keys():
                    eval_oct_performances[model_id] = {oct_num: eval_oct_avg_performance}
                else:
                    eval_oct_performances[model_id][oct_num] = eval_oct_avg_performance

            assert predict_extraction_total == eval_extraction_total == self._length

        assert -1 not in predict_oct_performance_array
        assert -1 not in eval_oct_performance_array

        if verbose:
            print('predict: \n', predict_oct_performance_array, '\n')
            print('eval: \n', eval_oct_performance_array, '\n')

        return predict_oct_performance_array, eval_oct_performance_array, predict_oct_performances, \
            eval_oct_performances

    @ staticmethod
    def _update_oct_performance_array(oct_performance_array, val, model_index, octant_index):
        oct_performance_array[model_index, octant_index] = val

    @staticmethod
    def _extract_oct_performance_val(oct_performance_array, model_index, octant_index):
        return oct_performance_array[model_index, octant_index]

    def _assign_best_models_to_octants(self):

        octant_model_map = {}

        for octant_index, oct_num in enumerate(self._oct_nums):

            assert octant_index == oct_num

            predict_performances = self._predict_model_oct_performance[:, octant_index]
            best_model_index = np.argmax(predict_performances)
            best_model_id = self._model_artifact_ids[int(best_model_index)]
            octant_model_map[oct_num] = (best_model_index, best_model_id)

        return octant_model_map

    def _composite_performance(self):

        predict_composite_performance = -1 * np.ones(self._length, dtype=np.float32)
        eval_composite_performance = -1 * np.ones(self._length, dtype=np.float32)

        predict_points_total = 0
        eval_points_total = 0

        for oct_num, (best_model_index, model_id) in self._octant_model_map.items():

            predict_extract_inds = self._get_extraction_indices(oct_num=oct_num, predict_eval_flag='predict')
            eval_extract_inds = self._get_extraction_indices(oct_num=oct_num, predict_eval_flag='eval')

            num_predict_points = len(predict_extract_inds[0])
            predict_points_total += num_predict_points
            num_eval_points = len(eval_extract_inds[0])
            eval_points_total += num_eval_points

            predict_performance_array = np.ravel(self._performance_arrays_predict[model_id])
            eval_performance_array = np.ravel(self._performance_arrays[model_id])

            predict_composite_performance[predict_extract_inds] = predict_performance_array[predict_extract_inds]
            eval_composite_performance[eval_extract_inds] = eval_performance_array[eval_extract_inds]

        assert -1 not in predict_composite_performance
        assert -1 not in eval_composite_performance

        predict_composite_performance = np.atleast_2d(predict_composite_performance).T
        eval_composite_performance = np.atleast_2d(eval_composite_performance).T

        return predict_composite_performance, eval_composite_performance

    def _make_extractor(self, oct_num, predict_eval_flag):

        if predict_eval_flag == 'predict':
            oct_labels = self._predict_oct_labels
        elif predict_eval_flag == 'eval':
            oct_labels = self._eval_oct_labels
        else:
            raise Exception("predict eval flag must be either 'predict' or 'eval'")

        extraction_indices = self._get_extraction_indices(oct_num=oct_num, predict_eval_flag=predict_eval_flag)
        extractor = np.zeros_like(oct_labels)
        extractor[extraction_indices] = 1

        return extractor

    def _get_extraction_indices(self, oct_num, predict_eval_flag):

        if predict_eval_flag == 'predict':
            oct_labels = self._predict_oct_labels
        elif predict_eval_flag == 'eval':
            oct_labels = self._eval_oct_labels
        else:
            raise Exception("predict eval flag must be either 'predict' or 'eval'")

        extraction_indices = np.where(oct_labels == oct_num)

        return extraction_indices

    def plot(self):

        predict_model_performances = []
        eval_model_performances = []
        ordered_model_ids = []

        for model_index, model_id in enumerate(self._model_artifact_ids):
            predict_model_oct_performance = self._predict_model_oct_performance[model_index, :]
            eval_model_oct_performance = self._eval_model_oct_performance[model_index, :]
            predict_model_performances.append(predict_model_oct_performance)
            eval_model_performances.append(eval_model_oct_performance)
            ordered_model_ids.append(model_id)

        plt.figure()
        for i, predict_model_oct_performance in enumerate(predict_model_performances):
            plt.plot(self._oct_nums, predict_model_oct_performance, label=ordered_model_ids[i])
        plt.xlabel('octant number')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('predict')
        plt.show()

        plt.figure()
        for i, eval_model_oct_performance in enumerate(eval_model_performances):
            plt.plot(self._oct_nums, eval_model_oct_performance, label=ordered_model_ids[i])
        plt.xlabel('octant number')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('eval')
        plt.show()

    def get_3d_distortion_perf_props(self, predict_eval_flag='eval', distortion_ids=None):
        """
        Shares name with equivalent methods in ModelDistortionPerformanceResult, CompositePerformanceResult, and
        ModelDistortionPerformanceOD but implemented differently
        """

        if predict_eval_flag in self._3d_distortion_perf_props.keys():
            return self._3d_distortion_perf_props[predict_eval_flag]

        if predict_eval_flag == 'predict':
            distortion_array = self._distortion_array_predict
            performance_array = self._predict_composite_performance
        elif predict_eval_flag == 'eval':
            distortion_array = self._distortion_array
            performance_array = self._eval_composite_performance
        else:
            raise ValueError("predict_eval_flag must be either 'predict' or 'eval'")

        res, blur, noise = distortion_array[:, 0], distortion_array[:, 1], distortion_array[:, 2]
        map_3d = build_3d_field(x=res, y=blur, z=noise, f=performance_array, data_dump=False)

        self._3d_distortion_perf_props[predict_eval_flag] = (self.res_vals, self.blur_vals, self.noise_vals,
                                                             map_3d, distortion_array, performance_array, None)

        self.archive_processed_props(predict_eval_flag=predict_eval_flag)
        print(f'archived processed {predict_eval_flag} performance props')

        return self._3d_distortion_perf_props[predict_eval_flag]


def get_composite_performance_result_od(config):

    performance_prediction_result_id_pairs = config['performance_prediction_result_id_pairs']
    performance_eval_result_id_pairs = config['performance_eval_result_ids_pairs']
    identifier = config['identifier']

    if 'res_boundary_weights' in config.keys():
        res_boundary_weights = config['res_boundary_weights']
    else:
        res_boundary_weights = None

    composite_performance_od = CompositePerformanceResultOD(
        performance_prediction_result_id_pairs=performance_prediction_result_id_pairs,
        performance_eval_result_id_pairs=performance_eval_result_id_pairs,
        identifier=identifier,
        res_boundary_weights=res_boundary_weights
    )

    composite_result_id = str(composite_performance_od)
    uid = composite_performance_od.uid
    output_dir = get_composite_performance_result_output_directory(composite_result_id, uid)

    return composite_performance_od, output_dir


def main(config):

    composite_performance_od, output_dir = get_composite_performance_result_od(config)
    
    flatten_axes = flatten_axes_from_cfg(config)
    flatten_axis_combinations = flatten_axis_combinations_from_cfg(config)
    fit_keys = fit_keys_from_cfg(config)

    if 'plot_together' in config.keys():
        plot_together = config['plot_together']
    else:
        plot_together = True 

    predict_data = composite_performance_od.get_3d_distortion_perf_props(predict_eval_flag='predict')
    eval_data = composite_performance_od.get_3d_distortion_perf_props(predict_eval_flag='eval')

    res_vals, blur_vals, noise_vals, map_3d_predict, predict_param_array, predict_perf_array, __ = predict_data
    __, __, __, map3d_measured, eval_parameter_array, eval_performance_array, __ = eval_data
    
    sub_dir, log_filename = get_sub_dir_and_log_filename(output_dir, '3d')
    add_bias_in_fits = False  # only applies to linear fits (all fits here nonlinear)

    with open(Path(sub_dir, log_filename), 'w') as output_file:

        for fit_key in fit_keys:

            fit_sub_dir, __ = get_sub_dir_and_log_filename(sub_dir, fit_key)
            fit_coefficients = fit(predict_param_array, predict_perf_array,
                                   distortion_ids=('res', 'blur', 'noise'),
                                   fit_key=fit_key,
                                   add_bias=False  # only applies to linear fits
                                   )
            eval_prediction = apply_fit(fit_coefficients,
                                        eval_parameter_array,
                                        distortion_ids=('res', 'blur', 'noise'),
                                        fit_key=fit_key,
                                        add_bias=False  # only applies to linear fits
                                        )
            eval_prediction_3d = build_3d_field(eval_parameter_array[:, 0],
                                                eval_parameter_array[:, 1],
                                                eval_parameter_array[:, 2],
                                                eval_prediction,
                                                data_dump=add_bias_in_fits
                                                )
            predict_fit_correlation = evaluate_fit(fit_coefficients,
                                                   predict_param_array,
                                                   predict_perf_array,
                                                   distortion_ids=('res', 'blur', 'noise'),
                                                   fit_key=fit_key,
                                                   add_bias=add_bias_in_fits,
                                                   )
            fit_correlation = evaluate_fit(fit_coefficients,
                                           eval_parameter_array,
                                           eval_performance_array,
                                           distortion_ids=('res', 'blur', 'noise'),
                                           fit_key=fit_key,
                                           add_bias=False,  # only applies to linear fits
                                           )

            aic_score = akaike_info_criterion(acc=eval_performance_array,
                                              n_trials=None,  # only used for binomial distributions
                                              acc_predicted=eval_prediction,
                                              num_parameters=len(fit_coefficients),
                                              distribution='normal'
                                              )
            
            plot.compare_2d_mean_views(f0=map3d_measured,
                                       f1=eval_prediction_3d,
                                       data_labels=('measured (eval)', 'fit'),
                                       x_vals=res_vals, y_vals=blur_vals, z_vals=noise_vals,
                                       distortion_ids=('res', 'blur', 'noise'),
                                       directory=fit_sub_dir,
                                       perf_metric='mAP',
                                       az_el_combinations='all',
                                       show_plots=False,
                                       flatten_axes=flatten_axes)

            plot.compare_1d_views(f0=map3d_measured,
                                  f1=eval_prediction_3d,
                                  data_labels=('measured (eval)', 'fit'),
                                  x_vals=res_vals, y_vals=blur_vals, z_vals=noise_vals,
                                  flatten_axis_combinations=flatten_axis_combinations,
                                  distortion_ids=('res', 'blur', 'noise'),
                                  directory=fit_sub_dir,
                                  result_id='3d_1d_projection',
                                  show_plots=False,
                                  ylabel='mAP',
                                  plot_together=plot_together)

            print(f'{fit_key} fit: \n', fit_coefficients, file=output_file)
            print(f'{fit_key} predict (direct) fit correlation: ', predict_fit_correlation,
                  file=output_file)
            print(f'{fit_key} eval fit correlation: ', fit_correlation,
                  file=output_file)
            print(f'{fit_key} fit aic score: ', aic_score, '\n',
                  file=output_file)

    return composite_performance_od


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'composite_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _composite_performance_od = main(run_config)
