import copy
import numpy as np
import argparse
from src.d04_analysis.distortion_performance import get_multiple_model_distortion_performance_results
from src.d04_analysis._shared_methods import _get_processed_instance_props_path, _check_extract_processed_props, \
    _archive_processed_props, _get_3d_distortion_perf_props
from src.d04_analysis.fit import nonlinear_fit, eval_nonlinear_fit
from src.d00_utils.definitions import STANDARD_COMPOSITE_DISTORTION_PERFORMANCE_PROPS_FILENAME, KEY_LENGTH, ROOT_DIR, \
    REL_PATHS
from src.d00_utils.functions import get_config
from src.d04_analysis.analysis_functions import conditional_mean_accuracy, build_3d_field
from pathlib import Path
from src.d04_analysis.distortion_performance import analyze_perf_1d, analyze_perf_2d, analyze_perf_3d


class CompositePerformanceResult(object):
    """
    Acts like the ModelDistortionPerformanceResult class, but contains the results of multiple models on two distorted
    image datasets, where the performance of the models on the frst dataset determines which model's result is used for
    each distortion point in the second dataset
    """

    def __init__(self, performance_prediction_result_ids, performance_eval_result_ids=None, identifier=None,
                 distortion_ids=('res', 'blur', 'noise'), surrogate_model_id='composite'):

        self.performance_prediction_result_ids = performance_prediction_result_ids
        self.eval_result_ids = performance_eval_result_ids
        self.distortion_ids = distortion_ids

        self.performance_prediction_results = get_multiple_model_distortion_performance_results(
            self.performance_prediction_result_ids, self.distortion_ids, make_dir=False, output_type='dict'
        )
        if self.eval_result_ids:
            self.eval_results = get_multiple_model_distortion_performance_results(
                self.eval_result_ids, self.distortion_ids, make_dir=False, output_type='dict'
            )
        else:
            self.eval_results = None

        self.result_id = self._get_composite_result_id()
        self.identifier = identifier
        if not self.identifier:
            self.identifier = self.result_id

        self.performance_prediction_dataset_id, self.dataset_id = self._screen_dataset_ids()

        self._res_perf_predict = None
        self._blur_perf_predict = None
        self._noise_perf_predict = None
        self.res = None  # eval
        self.blur = None  # eval
        self.noise = None  # eval
        self.res_vals = None  # _vals are the unique distortion values of each distortion type (common across datasets)
        self.blur_vals = None
        self.noise_vals = None
        self._get_distortion_space()  # assigns distortion vectors to distortion variables above

        if self.eval_results:
            self.labels = next(iter(self.eval_results.values())).labels
        else:
            self.labels = None
            self.predicts = None  # eval predicts

        self._distortion_pt_perf_predict_hashes, self._distortion_pt_eval_hashes = self._assign_distortion_pt_hashes()
        self.model_ids, self.model_id_ppr_id_pairs = self._get_model_id_performance_prediction_result_pairs()
        self.perf_predict_top_1_array, self.perf_predict_prediction_array = self._make_perf_predict_arrays()
        self._model_id_to_eval_id_map = self._map_model_ids_to_eval_result_ids()

        self._model_map = self._assign_models()
        self._hash_to_row_idx_map = self._get_hash_to_row_idx_map()

        # self.filtered_predict_top_1_vec and self.filtered_perf_predict_predicts the top 1 vector and predict vector
        # respectively that result from applying self._model_id_to_eval_id_map to the performance prediction array
        self.filtered_predict_top_1_vec, self.filtered_perf_predict_predicts = self._make_perf_predict_arrays()

        if self.eval_results:
            self.eval_top_1_array, self.eval_prediction_array = self._make_eval_arrays()
            self.top_1_vec, self.predicts = self.eval_performance()  # eval predicts
        else:
            self.eval_top_1_array = None
            self.top_1_vec = None

        # other attributes needed for compatibility with functions that use ModelDistortionPerformance instances
        if self.top_1_vec is not None:
            self.instance_hash = hash(tuple(self.top_1_vec))
        else:
            self.instance_hash = hash(tuple(self.performance_prediction_result_ids))

        self.distortions = {
            'res': self.res,
            'blur': self.blur,
            'noise': self.noise,

            '_res_perf_predict': self._res_perf_predict,
            '_blur_perf_predict': self._blur_perf_predict,
            '_noise_perf_predict': self._noise_perf_predict,
        }

        self.model_id = surrogate_model_id

        self.perf_prediction_fit = None

    # *** methods needed for compatibility with functions that use ModelDistortionPerformance instances ***************
    def get_processed_instance_props_path(self):
        return _get_processed_instance_props_path(self)

    def check_extract_processed_props(self):
        return _check_extract_processed_props(self)

    def archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                perf_array):
        return _archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                        perf_array)

    def get_3d_distortion_perf_props(self, distortion_ids):
        return _get_3d_distortion_perf_props(self, distortion_ids)

    def conditional_accuracy(self, distortion_id, per_class=False):
        assert per_class is False
        return conditional_mean_accuracy(self.labels, self.predicts, self.distortions[distortion_id],
                                         per_class=per_class)  # eval predicts

    # *****************************************************************************************************************

    def __len__(self):
        if self.eval_results:
            return len(self._distortion_pt_eval_hashes)
        else:
            return len(self._distortion_pt_perf_predict_hashes)

    def __str__(self):
        if self.identifier is not None:
            return self.identifier
        else:
            return self.result_id

    def _get_composite_result_id(self):

        composite_result_id = 'cr'
        for performance_prediction_result_id in self.performance_prediction_result_ids:
            composite_result_id = f'{composite_result_id}-{performance_prediction_result_id[:KEY_LENGTH]}'

        if self.eval_results:
            composite_result_id = f'{composite_result_id}__eval'
            for eval_result_id in self.eval_result_ids:
                composite_result_id = f'{composite_result_id}-{eval_result_id[:4]}'

        return composite_result_id

    def _get_composite_model_id(self):
        pass

    def _screen_dataset_ids(self):
        """
        checks that all performance prediction results of each type come from the same underlying test dataset AND
        checks that the performance prediction and eval dataset ids are not identical to each other
        """

        performance_prediction_dataset_id = None
        eval_dataset_id = None

        for result_id, result in self.performance_prediction_results.items():
            if not performance_prediction_dataset_id:
                performance_prediction_dataset_id = result.dataset_id
            elif performance_prediction_dataset_id != result.dataset_id:
                raise Exception(f'performance prediction dataset ids inconsistent, {performance_prediction_dataset_id} '
                                f'!= {result.dataset_id}')

        if self.eval_results:
            for result_id, result in self.eval_results.items():
                if not eval_dataset_id:
                    eval_dataset_id = result.dataset_id
                elif eval_dataset_id != result.dataset_id:
                    raise Exception(f'eval dataset ids inconsistent, {eval_dataset_id} != '
                                    f'{result.dataset_id}')

        if eval_dataset_id == performance_prediction_dataset_id:
            raise Exception('eval dataset id and performance prediction dataset ids must be different')

        return performance_prediction_dataset_id, eval_dataset_id

    def _get_distortion_space(self):

        predictive_result = next(iter(self.performance_prediction_results.values()))
        res_p, blur_p, noise_p = predictive_result.res, predictive_result.blur, predictive_result.noise

        self.res_vals = np.unique(res_p)
        self.blur_vals = np.unique(blur_p)
        self.noise_vals = np.unique(noise_p)

        if self.eval_results:
            eval_result = next(iter(self.eval_results.values()))
            res_e, blur_e, noise_e = eval_result.res, eval_result.blur, eval_result.noise

            if not np.array_equal(self.res_vals, np.unique(res_e)):
                raise Exception('res distortion spaces unmatched')
            if not np.array_equal(self.blur_vals, np.unique(blur_e)):
                raise Exception('blur distortion spaces unmatched')
            if not np.array_equal(self.noise_vals, np.unique(noise_e)):
                raise Exception('noise distortion spaces unmatched')

            self.res = res_e
            self.blur = blur_e
            self.noise = noise_e

        self._res_perf_predict = res_p
        self._blur_perf_predict = blur_p
        self._noise_perf_predict = noise_p

    def _assign_distortion_pt_hashes(self):

        """
        Generates arrays containing the hash value of each unique distortion distortion point (res, blur, noise)
        for the combination of distortions applied to each performance predict and performance eval image.
        """

        distortion_array_p = np.stack([self._res_perf_predict, self._blur_perf_predict, self._noise_perf_predict],
                                      axis=0)
        predict_hashes = [hash(tuple(distortion_array_p[:, i])) for i in range(len(distortion_array_p[0, :]))]
        predict_hashes = np.asarray(predict_hashes)
        if self.eval_results:
            distortion_array_e = np.stack([self.res, self.blur, self.noise], axis=0)
            eval_hashes = [hash(tuple(distortion_array_e[:, i])) for i in range(len(distortion_array_e[0, :]))]
            eval_hashes = np.asarray(eval_hashes)
            assert set(predict_hashes) == set(eval_hashes)
        else:
            eval_hashes = None

        return predict_hashes, eval_hashes

    def _assign_models(self):
        """
        Finds the best performing model at each distortion point and maps it to the associated hash value for that
        distortion point.
        """
        model_map = {}

        for hash_val in set(self._distortion_pt_perf_predict_hashes):
            sub_array = self.perf_predict_top_1_array[:, np.where(
                self._distortion_pt_perf_predict_hashes == hash_val)[0]]
            model_accuracies = np.mean(sub_array, axis=1)
            best_model_id = self.model_ids[np.argmax(model_accuracies)]
            model_map[hash_val] = best_model_id

        return model_map

    def _get_model_id_performance_prediction_result_pairs(self):
        model_id_ppr_id_pairs = []
        model_ids = []
        for ppr_id, ppr in self.performance_prediction_results.items():
            model_id = ppr.model_id
            if model_id in model_ids:
                raise Exception(f'duplicate model id {model_id} found')
            model_ids.append(model_id)
            model_id_ppr_id_pairs.append((model_id, ppr_id))
        model_id_ppr_id_pairs = tuple(model_id_ppr_id_pairs)

        if self.eval_results:
            for er_id, er in self.eval_results.items():
                if er.model_id not in model_ids:
                    raise Exception(f'eval_results contains test result {er_id} from model {er.model_id} not included '
                                    f'in performance prediction results')

        model_ids = tuple(model_ids)  # convert to tuple for immutability
        model_id_ppr_id_pairs = tuple(model_id_ppr_id_pairs) # convert to tuple for immutability

        return model_ids, model_id_ppr_id_pairs

    def _map_model_ids_to_eval_result_ids(self):
        """
        Creates a dictionary linking model ids to their associated evaluation result id
        """
        model_id_to_eval_id_map = {}
        for eval_result_id, eval_result in self.eval_results.items():
            model_id = eval_result.model_id
            if model_id in model_id_to_eval_id_map.keys():
                raise Exception(f'model {model_id} repeated in eval_results')
            model_id_to_eval_id_map[model_id] = eval_result_id

        return model_id_to_eval_id_map

    def _make_perf_predict_arrays(self):
        """
        Creates arrays in which the rows are performance prediction result top 1 accuracy vectors and top 1 predict
        vectors
        """
        top_1_array = np.zeros(
            (len(self.performance_prediction_results), len(self._distortion_pt_perf_predict_hashes)))
        predict_array = np.zeros_like(top_1_array)

        for i, (model_id, ppr_id) in enumerate(self.model_id_ppr_id_pairs):
            ppr = self.performance_prediction_results[ppr_id]
            top_1_vec = ppr.top_1_vec
            predict_vec = ppr.predicts
            top_1_array[i, :] = top_1_vec
            predict_array[i, :] = predict_vec
        return top_1_array, predict_array

    def _make_eval_arrays(self):
        """
        Creates arrays in which the rows are performance eval result top 1 accuracy vectors and top 1 predict
        vectors
        """
        top_1_array = np.zeros(
            (len(self.eval_results), len(self._distortion_pt_eval_hashes)))
        predict_array = np.zeros_like(top_1_array)

        for i, (model_id, __) in enumerate(self.model_id_ppr_id_pairs):
            eval_result_id = self._model_id_to_eval_id_map[model_id]
            eval_result = self.eval_results[eval_result_id]
            top_1_vec = eval_result.top_1_vec
            predict_vec = eval_result.predicts
            top_1_array[i, :] = top_1_vec
            predict_array[i, :] = predict_vec

        return top_1_array, predict_array

    def _get_hash_to_row_idx_map(self):
        """
        Creates a dictionary linking each distortion point's hash value to the row index of is best performing model.
        The CompositePerformanceResult class uses the ordering of self.model_id_ppr_id_pairs to ensure that each model
        is assigned to the same row in the top 1 accuracy and predict arrays for both performance prediction and eval
        """
        hash_to_row_idx_map = {}
        for hash_val, model_id in self._model_map.items():
            hash_to_row_idx_map[hash_val] = self.model_ids.index(model_id)

        return hash_to_row_idx_map

    def _max_performance_predict(self):
        """
        Returns a composite top 1 vector consisting of the best possible performance prediction
        """
        return np.max(self.perf_predict_top_1_array, axis=0)

    def eval_performance(self):
        """
        Uses the mapping of each distortion point hash value to the associated row index of the best performing model
        at that distortion point to extract the associated eval_top_1_array and eval_prediction_array values, returning
        the composite top 1 vector (top_1_vec) and composite prediction vector (predict vec)
        """
        top_1_vec = -1 * np.ones(np.shape(self.eval_top_1_array)[1])
        predict_vec = -1 * np.ones_like(top_1_vec)
        for hash_val in self._hash_to_row_idx_map.keys():
            column_indices = np.where(self._distortion_pt_eval_hashes == hash_val)[0]
            row_index = self._hash_to_row_idx_map[hash_val]

            top_1_vec[column_indices] = self.eval_top_1_array[row_index, column_indices]
            predict_vec[column_indices] = self.eval_prediction_array[row_index, column_indices]

        assert -1 not in top_1_vec  # make sure every value has been written
        assert -1 not in predict_vec

        return top_1_vec, predict_vec

    def eval_performance_predict(self):
        """
        Applies the same mapping used in self.eval_performance() to generate the equivalent top 1 and predict vectors
        for the performance prediction results. Effectively gets the composite performance prediction that result from
        picking the best performing model on the performance prediction dataset (akin to training accuracy)
        """
        top_1_vec = -1 * np.ones(np.shape(self.perf_predict_top_1_array)[1])
        predict_vec = -1 * np.ones_like(top_1_vec)
        for hash_val in self._hash_to_row_idx_map.keys():
            column_indices = np.where(self._distortion_pt_eval_hashes == hash_val)[0]
            row_index = self._hash_to_row_idx_map[hash_val]

            top_1_vec[column_indices] = self.perf_predict_top_1_array[row_index, column_indices]
            predict_vec[column_indices] = self.perf_predict_prediction_array[row_index, column_indices]

        assert -1 not in top_1_vec  # make sure every value has been written
        assert -1 not in predict_vec

        return top_1_vec, predict_vec

    def eval_performance_cheating(self):

        pass

    def fit(self, add_bias=True):

        x_vals, y_vals, z_vals, perf_3d, distortion_array, perf_array = self.get_3d_distortion_perf_props(
            distortion_ids=self.distortion_ids)
        w = nonlinear_fit(distortion_array, perf_array, distortion_ids=self.distortion_ids, add_bias=add_bias)
        self.perf_prediction_fit = w

    def run_performance_prediction(self):
        if self.perf_prediction_fit is None:
            self.fit()


def get_composite_performance_result(performance_prediction_result_ids=None, performance_eval_result_ids=None,
                                     identifier=None, parent_dir='default', config=None, overwrite_extracted_props=True,
                                     make_dir=True):
    if config:
        performance_prediction_result_ids = config['performance_prediction_result_ids']
        performance_eval_result_ids = config['performance_eval_result_ids']
        identifier = config['identifier']
        overwrite_extracted_props = config['overwrite_extracted_props']

    composite_performance_result = CompositePerformanceResult(
        performance_prediction_result_ids, performance_eval_result_ids=performance_eval_result_ids,
        identifier=identifier)

    composite_result_id = composite_performance_result.result_id
    output_dir = Path(ROOT_DIR, REL_PATHS['composite_performance'], composite_result_id)

    if make_dir and not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    return composite_performance_result, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='composite_distortion_analysis_config.yml',
                        help='config filename to be used')
    # parser.add_argument('--config_name', default='pl_abbrev_oct_composite_config.yml',
    #                     help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'composite_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _composite_performance, _output_dir = get_composite_performance_result(config=run_config)
    with open(Path(_output_dir, 'test_composite_result_log.txt'), 'w') as output_file:
        # analyze_perf_1d(_composite_performance, log_file=output_file, directory=_output_dir, per_class=False,
        #                 distortion_ids=('res', 'blur', 'noise'))
        # analyze_perf_2d(_composite_performance, log_file=output_file, directory=_output_dir)
        analyze_perf_3d(_composite_performance, log_file=output_file, directory=_output_dir)
