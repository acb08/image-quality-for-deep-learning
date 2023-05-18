from src.obj_det_analysis.compare_results import get_multiple_od_distortion_performance_results
from src.utils.functions import get_config
import numpy as np
import argparse
from pathlib import Path
from src.utils.definitions import DISTORTION_RANGE_90
import matplotlib.pyplot as plt


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
                 distortion_ids=('res', 'blur', 'noise')):

        self.performance_prediction_result_id_pairs = performance_prediction_result_id_pairs
        self.performance_eval_result_id_pairs = performance_eval_result_id_pairs
        self.identifier = identifier
        self.distortion_ids = distortion_ids

        _predict_data = load_multiple_3d_perf_results(self.performance_prediction_result_id_pairs,
                                                      self.distortion_ids)

        self._perf_3d_predict = _predict_data[3]  # perf_results_3d, parameter_array, perf_arrays
        self.distortion_array_predict = _predict_data[4]
        self.performance_arrays_predict = _predict_data[5]
        self.model_artifact_id_set = _predict_data[6]
        self.length = _predict_data[7]
        del _predict_data

        _eval_data = load_multiple_3d_perf_results(self.performance_eval_result_id_pairs,
                                                   self.distortion_ids)

        self.res_vals = _eval_data[0]
        self.blur_vals = _eval_data[1]
        self.noise_vals = _eval_data[2]
        self._perf_3d_eval = _eval_data[3]
        self.distortion_array = _eval_data[4]
        self.performance_arrays = _eval_data[5]
        _eval_model_artifact_ids = _eval_data[6]
        _eval_length = _eval_data[7]
        del _eval_data

        assert self.performance_arrays.keys() == self.performance_arrays_predict.keys()
        assert _eval_model_artifact_ids == self.model_artifact_id_set
        assert set(self.performance_arrays.keys()) == self.model_artifact_id_set
        assert self.length == _eval_length

        self.model_artifact_ids = tuple(self.model_artifact_id_set)
        del self.model_artifact_id_set  # ensure that model id order is constant by using self.model_artifact_ids tuple

        self.res_boundary, self.blur_boundary, self.noise_boundary = self.get_octant_boundaries()
        self.predict_oct_vectors, self.oct_vectors = self.assign_octant_vectors()

        self.predict_oct_labels = label_octant_array(self.predict_oct_vectors)
        self.oct_labels = label_octant_array(self.oct_vectors)

        self.oct_nums = np.unique(self.oct_labels)
        assert np.array_equal(self.oct_nums, np.unique(self.predict_oct_labels))
        self.oct_nums = tuple(self.oct_nums)  # make immutable just to be safe

        (self.predict_model_oct_performance, self.eval_model_oct_performance,  self.predict_model_oct_performance_dict,
         self.eval_model_oct_performance_dict) = self.get_performance_by_octant()
        self.octant_model_map = self.assign_best_models_to_octants()

        self.predict_composite_performance, self.eval_composite_performance = self.composite_performance()

        self._distortion_perf_props_3d = {}

    def __len__(self):
        return self.length

    def get_octant_boundaries(self):

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

        res_boundary = (4 * res_min + res_max) / 5  # TODO: put back to simple midpoint
        blur_boundary = (blur_min + blur_max) / 2
        noise_boundary = (noise_min + noise_max) / 2

        return res_boundary, blur_boundary, noise_boundary

    def assign_octant_vectors(self):

        predict_oct_ids = -1 * np.ones_like(self.distortion_array_predict)

        res_predict = self.distortion_array_predict[:, 0]
        blur_predict = self.distortion_array_predict[:, 1]
        noise_predict = self.distortion_array_predict[:, 2]

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

        eval_oct_ids = -1 * np.ones_like(self.distortion_array)
        res = self.distortion_array[:, 0]
        blur = self.distortion_array[:, 1]
        noise = self.distortion_array[:, 2]

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

    def get_performance_by_octant(self, verbose=True):

        predict_oct_performances = {}
        eval_oct_performances = {}

        num_models = len(self.model_artifact_ids)
        num_octants = len(self.oct_nums)
        assert num_octants == 8

        predict_oct_performance_array = -1 * np.ones((num_models, num_octants))
        eval_oct_performance_array = -1 * np.ones((num_models, num_octants))

        for model_index, model_id in enumerate(self.model_artifact_ids):

            predict_performance_array = np.ravel(self.performance_arrays_predict[model_id])
            eval_performance_array = np.ravel(self.performance_arrays[model_id])
            predict_extraction_total = 0
            eval_extraction_total = 0

            for octant_index, oct_num in enumerate(self.oct_nums):

                assert octant_index == oct_num

                _predict_extractor = np.zeros_like(self.predict_oct_labels)
                _predict_extractor[np.where(self.predict_oct_labels == oct_num)] = 1
                predict_extractor = self.make_extractor(oct_num=oct_num, predict_eval_flag='predict')
                assert np.array_equal(_predict_extractor, predict_extractor)

                num_predict_points = np.sum(predict_extractor)
                predict_extraction_total += num_predict_points

                predict_oct_avg_performance = np.sum(predict_extractor * predict_performance_array) / num_predict_points
                self.update_oct_performance_array(
                    oct_performance_array=predict_oct_performance_array,
                    val=predict_oct_avg_performance,
                    model_index=model_index,
                    octant_index=octant_index
                )

                if model_id not in predict_oct_performances.keys():
                    predict_oct_performances[model_id] = {oct_num: predict_oct_avg_performance}
                else:
                    predict_oct_performances[model_id][oct_num] = predict_oct_avg_performance

                _eval_extractor = np.zeros_like(self.oct_labels)
                _eval_extractor[np.where(self.predict_oct_labels == oct_num)] = 1
                eval_extractor = self.make_extractor(oct_num=oct_num, predict_eval_flag='eval')
                assert np.array_equal(_eval_extractor, eval_extractor)

                num_eval_points = np.sum(eval_extractor)
                eval_extraction_total += num_eval_points

                eval_oct_avg_performance = np.sum(eval_extractor * eval_performance_array) / num_eval_points
                self.update_oct_performance_array(
                    oct_performance_array=eval_oct_performance_array,
                    val=eval_oct_avg_performance,
                    model_index=model_index,
                    octant_index=octant_index
                )

                if model_id not in eval_oct_performances.keys():
                    eval_oct_performances[model_id] = {oct_num: eval_oct_avg_performance}
                else:
                    eval_oct_performances[model_id][oct_num] = eval_oct_avg_performance

            assert predict_extraction_total == eval_extraction_total == self.length

        assert -1 not in predict_oct_performance_array
        assert -1 not in eval_oct_performance_array

        if verbose:
            print('predict: \n', predict_oct_performance_array, '\n')
            print('eval: \n', eval_oct_performance_array, '\n')

        return predict_oct_performance_array, eval_oct_performance_array, predict_oct_performances, \
            eval_oct_performances

    @ staticmethod
    def update_oct_performance_array(oct_performance_array, val, model_index, octant_index):
        oct_performance_array[model_index, octant_index] = val

    @staticmethod
    def extract_oct_performance_val(oct_performance_array, model_index, octant_index):
        return oct_performance_array[model_index, octant_index]

    def assign_best_models_to_octants(self):

        octant_model_map = {}

        for octant_index, oct_num in enumerate(self.oct_nums):

            assert octant_index == oct_num

            predict_performances = self.predict_model_oct_performance[:, octant_index]
            best_model_index = np.argmax(predict_performances)
            best_model_id = self.model_artifact_ids[int(best_model_index)]
            octant_model_map[oct_num] = (best_model_index, best_model_id)

        return octant_model_map

    def composite_performance(self):

        predict_composite_performance = -1 * np.ones(self.length)
        eval_composite_performance = -1 * np.ones(self.length)

        predict_points_total = 0
        eval_points_total = 0

        for oct_num, (best_model_index, model_id) in self.octant_model_map.items():

            predict_extract_inds = self.get_extraction_indices(oct_num=oct_num, predict_eval_flag='predict')
            eval_extract_inds = self.get_extraction_indices(oct_num=oct_num, predict_eval_flag='eval')

            num_predict_points = len(predict_extract_inds[0])
            predict_points_total += num_predict_points
            num_eval_points = len(eval_extract_inds[0])
            eval_points_total += num_eval_points

            predict_performance_array = np.ravel(self.performance_arrays_predict[model_id])
            eval_performance_array = np.ravel(self.performance_arrays[model_id])

            predict_composite_performance[predict_extract_inds] = predict_performance_array[predict_extract_inds]
            eval_composite_performance[eval_extract_inds] = eval_performance_array[eval_extract_inds]

        assert -1 not in predict_composite_performance
        assert -1 not in eval_composite_performance

        return predict_composite_performance, eval_composite_performance

    def make_extractor(self, oct_num, predict_eval_flag):

        if predict_eval_flag == 'predict':
            oct_labels = self.predict_oct_labels
        elif predict_eval_flag == 'eval':
            oct_labels = self.oct_labels
        else:
            raise Exception("predict eval flag must be either 'predict' or 'eval'")

        extraction_indices = self.get_extraction_indices(oct_num=oct_num, predict_eval_flag=predict_eval_flag)
        extractor = np.zeros_like(oct_labels)
        extractor[extraction_indices] = 1

        return extractor

    def get_extraction_indices(self, oct_num, predict_eval_flag):

        if predict_eval_flag == 'predict':
            oct_labels = self.predict_oct_labels
        elif predict_eval_flag == 'eval':
            oct_labels = self.oct_labels
        else:
            raise Exception("predict eval flag must be either 'predict' or 'eval'")

        extraction_indices = np.where(oct_labels == oct_num)

        return extraction_indices

    def plot(self):

        predict_model_performances = []
        eval_model_performances = []
        ordered_model_ids = []

        for model_index, model_id in enumerate(self.model_artifact_ids):
            predict_model_oct_performance = self.predict_model_oct_performance[model_index, :]
            eval_model_oct_performance = self.eval_model_oct_performance[model_index, :]
            predict_model_performances.append(predict_model_oct_performance)
            eval_model_performances.append(eval_model_oct_performance)
            ordered_model_ids.append(model_id)

        plt.figure()
        for i, predict_model_oct_performance in enumerate(predict_model_performances):
            plt.plot(self.oct_nums, predict_model_oct_performance, label=ordered_model_ids[i])
        plt.xlabel('octant number')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('predict')
        plt.show()

        plt.figure()
        for i, eval_model_oct_performance in enumerate(eval_model_performances):
            plt.plot(self.oct_nums, eval_model_oct_performance, label=ordered_model_ids[i])
        plt.xlabel('octant number')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('eval')
        plt.show()


def get_composite_performance_result_od(config):

    performance_prediction_result_id_pairs = config['performance_prediction_result_id_pairs']
    performance_eval_result_id_pairs = config['performance_eval_result_ids_pairs']
    identifier = config['identifier']

    composite_performance_od = CompositePerformanceResultOD(
        performance_prediction_result_id_pairs=performance_prediction_result_id_pairs,
        performance_eval_result_id_pairs=performance_eval_result_id_pairs,
        identifier=identifier
    )

    return composite_performance_od


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'composite_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _c_perf_od = get_composite_performance_result_od(run_config)
