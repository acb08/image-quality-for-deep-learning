import json
import numpy as np
import argparse
from src.d04_analysis.distortion_performance import get_multiple_model_distortion_performance_results
from src.d04_analysis._shared_methods import _get_processed_instance_props_path, _check_extract_processed_props, \
    _archive_processed_props, _get_3d_distortion_perf_props, get_instance_hash
from src.d00_utils.definitions import STANDARD_UID_FILENAME, KEY_LENGTH, ROOT_DIR, \
    REL_PATHS, PROJECT_ID, DISTORTION_RANGE
from src.d00_utils.functions import get_config, log_config, increment_suffix
from src.d04_analysis.analysis_functions import conditional_mean_accuracy, get_sub_dir_and_log_filename
from pathlib import Path
from src.d04_analysis.distortion_performance import analyze_perf_1d, analyze_perf_2d, analyze_perf_3d, \
    performance_fit_summary_text_dump
from hashlib import blake2b
import copy


class CompositePerformanceResult(object):
    """
    Acts like the ModelDistortionPerformanceResult class, but contains the results of multiple models on two distorted
    image datasets, where the performance of the models on the frst dataset determines which model's result is used for
    each distortion point in the second dataset
    """

    def __init__(self, performance_prediction_result_ids, assign_by_octant_only=False, performance_eval_result_ids=None,
                 identifier=None, distortion_ids=('res', 'blur', 'noise'), surrogate_model_id='composite'):

        self.performance_prediction_result_ids = performance_prediction_result_ids
        self.assign_by_octant_only = assign_by_octant_only
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
        self.uid = get_composite_result_uid(self.performance_prediction_result_ids,
                                            self.eval_result_ids)
        self.identifier = identifier

        if not self.identifier:
            self.identifier = self.result_id

        # if self.assign_by_octant_only and self.result_id != self.identifier:
        #     self.identifier = f'{self.identifier}-om'

        self.performance_prediction_dataset_id, self.dataset_id = self._screen_dataset_ids()

        self.res_predict = None
        self.blur_predict = None
        self.noise_predict = None
        self.res = None  # eval
        self.blur = None  # eval
        self.noise = None  # eval
        self.res_vals = None  # _vals are the unique distortion values of each distortion type (common across datasets)
        self.blur_vals = None
        self.noise_vals = None
        self._get_distortion_space()  # assigns distortion vectors to distortion variables above

        self._distortion_perf_props_3d = {}  # used to store the output of self.get_3d_distortion_perf_props()

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

        # self.top_1_vec_predict and self.filtered_perf_predict_predicts are the top 1 vector and predict vector
        # respectively that result from applying self._model_id_to_eval_id_map to the performance prediction array
        self.top_1_vec_predict, self.filtered_perf_predict_predicts = self.eval_performance_predict()

        if self.eval_results:
            self.eval_top_1_array, self.eval_prediction_array = self._make_eval_arrays()
            self.top_1_vec, self.predicts = self.eval_performance()  # eval predicts
        else:
            self.eval_top_1_array = None
            self.top_1_vec = None

        # other attributes needed for compatibility with functions that use ModelDistortionPerformance instances
        self.instance_hashes = {'predict': get_instance_hash(self.top_1_vec_predict,
                                                             self.noise_predict)}
        if self.top_1_vec is not None:
            self.instance_hashes['eval'] = get_instance_hash(self.top_1_vec, self.noise)

        # structure below allows access of the form model_performance.distortions[predict_eval_flag][distortion_id]
        _predict_distortions = {
            'res': self.res_predict,
            'blur': self.blur_predict,
            'noise': self.noise_predict
        }
        _eval_distortions = {
            'res': self.res,
            'blur': self.blur,
            'noise': self.noise
        }
        self.distortions = {
            'res': self.res,
            'blur': self.blur,
            'noise': self.noise,
            'predict': _predict_distortions,
            'eval': _eval_distortions
        }

        self.model_id = surrogate_model_id
        self.perf_prediction_fit = None  # tuple of form (fit_coefficients, fit_key) once assigned

    def mean_accuracy(self):
        if self.top_1_vec is not None:
            return np.mean(self.top_1_vec)
        else:
            return None

    def mean_accuracy_predict(self):
        return np.mean(self.top_1_vec_predict)

    # *** methods needed for compatibility with functions that use ModelDistortionPerformance instances ***************
    def get_processed_instance_props_path(self, predict_eval_flag):
        return _get_processed_instance_props_path(self, predict_eval_flag=predict_eval_flag)

    def check_extract_processed_props(self, predict_eval_flag):
        return _check_extract_processed_props(self, predict_eval_flag=predict_eval_flag)

    def archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                perf_array, predict_eval_flag):
        return _archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                        perf_array, predict_eval_flag=predict_eval_flag)

    def get_3d_distortion_perf_props(self, distortion_ids, predict_eval_flag='eval'):
        """
        returns 3d distortion performance properties in the form of a tuple:
            res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, None

        Also stores the tuple above in the dictionary self.distortion_perf_props_3d, where the key is predict_eval_flag
        so that any subsequent calls can avoid re-computing the distortion performance properties.
        """
        if predict_eval_flag in self._distortion_perf_props_3d:
            return self._distortion_perf_props_3d[predict_eval_flag]
        else:
            self._distortion_perf_props_3d[predict_eval_flag] = _get_3d_distortion_perf_props(
                self, distortion_ids, predict_eval_flag=predict_eval_flag)
        return self._distortion_perf_props_3d[predict_eval_flag]

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

    def __repr__(self):
        return self.result_id

    def _get_composite_result_id(self):

        composite_result_id = 'cr'
        for performance_prediction_result_id in self.performance_prediction_result_ids:
            composite_result_id = f'{composite_result_id}-{performance_prediction_result_id[:KEY_LENGTH]}'

        if self.eval_results:
            composite_result_id = f'{composite_result_id}__eval'
            for eval_result_id in self.eval_result_ids:
                composite_result_id = f'{composite_result_id}-{eval_result_id[:4]}'

        # if self.assign_by_octant_only:
        #     composite_result_id = f'{composite_result_id}-om'

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

        self.res_predict = res_p
        self.blur_predict = blur_p
        self.noise_predict = noise_p

    def _assign_distortion_pt_hashes(self,):

        """
        Generates arrays containing the hash value of each unique distortion distortion point (res, blur, noise)
        for the combination of distortions applied to each performance predict and performance eval image.

        Note: these hash values will change from run to run due to a random seed in the underlying hash function. They
        should not be saved for later user without recognizing that computing hashes for comparison after the fact
        will fail.
        """

        distortion_array_p = np.stack([self.res_predict, self.blur_predict, self.noise_predict],
                                      axis=0)

        if self.assign_by_octant_only:
            distortion_array_p = assign_octant_labels(distortion_array_p)

        predict_hashes = [hash(tuple(distortion_array_p[:, i])) for i in range(len(distortion_array_p[0, :]))]
        predict_hashes = np.asarray(predict_hashes)
        if self.eval_results:
            distortion_array_e = np.stack([self.res, self.blur, self.noise], axis=0)
            if self.assign_by_octant_only:
                distortion_array_e = assign_octant_labels(distortion_array_e)
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
            best_model_id = self.model_ids[int(np.argmax(model_accuracies))]
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
        model_id_ppr_id_pairs = tuple(model_id_ppr_id_pairs)  # convert to tuple for immutability

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
            column_indices = np.where(self._distortion_pt_perf_predict_hashes == hash_val)[0]
            row_index = self._hash_to_row_idx_map[hash_val]

            top_1_vec[column_indices] = self.perf_predict_top_1_array[row_index, column_indices]
            predict_vec[column_indices] = self.perf_predict_prediction_array[row_index, column_indices]

        assert -1 not in top_1_vec  # make sure every value has been written
        assert -1 not in predict_vec

        return top_1_vec, predict_vec

    def eval_performance_cheating(self):

        pass

    def compare_predict_eval(self, distortion_ids=('res', 'blur', 'noise')):
        __, __, __, perf_3d_predict, __, perf_array_predict, __ = self.get_3d_distortion_perf_props(
            distortion_ids=distortion_ids, predict_eval_flag='predict')
        __, __, __, perf_3d_eval, __, perf_array_eval, __ = self.get_3d_distortion_perf_props(
            distortion_ids=distortion_ids, predict_eval_flag='eval')

        diff_3d = perf_3d_eval - perf_3d_predict
        diff_vec = perf_array_predict - perf_array_eval
        return diff_3d, diff_vec


def assign_octant_labels(distortion_array):
    """
    Requires a 3 x N distortion array where distortion_array[0, :] = res, distortion_array[1, :] = blur, and
    distortion_array[2, :] = noise.
    """
    distortion_range = DISTORTION_RANGE[PROJECT_ID]

    res = distortion_array[0, :]
    blur = distortion_array[1, :]
    noise = distortion_array[2, :]

    res_assignment = -1 * np.ones_like(res)
    blur_assignment = -1 * np.ones_like(blur)
    noise_assignment = -1 * np.ones_like(noise)

    r0, r1 = distortion_range['res']
    r0 = r0 / max(distortion_range['res'])  # necessary because sat6 res values specified in pixels in DISTORTION_RANGE
    r1 = r1 / max(distortion_range['res'])  # has no effect for places365 since max(distortion_range['places365'] = 1
    res_boundary = (r0 + r1) / 2
    res_assignment[np.where(res > res_boundary)] = 0  # note that res label 0 corresponds to higher res values,
    res_assignment[np.where(res <= res_boundary)] = 1  # in contrast to blur and noise where large values assigned 1
    assert -1 not in res_assignment

    __, b0, b1 = distortion_range['blur']
    blur_boundary = (b0 + b1) / 2
    blur_assignment[np.where(blur < blur_boundary)] = 0
    blur_assignment[np.where(blur >= blur_boundary)] = 1
    assert -1 not in blur_assignment

    n0, n1 = distortion_range['noise']
    noise_boundary = (n0 + n1) / 2
    noise_assignment[np.where(noise < noise_boundary)] = 0
    noise_assignment[np.where(noise >= noise_boundary)] = 1
    assert -1 not in noise_assignment

    assignment_array = np.stack([res_assignment, blur_assignment, noise_assignment], axis=0)

    return assignment_array


def get_composite_performance_result(performance_prediction_result_ids=None, performance_eval_result_ids=None,
                                     identifier=None, config=None, suffix=None, assign_by_octant_only=True):
    if config:
        performance_prediction_result_ids = config['performance_prediction_result_ids']
        performance_eval_result_ids = config['performance_eval_result_ids']
        identifier = config['identifier']
        try:
            assign_by_octant_only = config['assign_by_octant_only']
        except KeyError:
            pass
        # overwrite_extracted_props = config['overwrite_extracted_props']

    composite_performance_result = CompositePerformanceResult(
        performance_prediction_result_ids, assign_by_octant_only=assign_by_octant_only,
        performance_eval_result_ids=performance_eval_result_ids, identifier=identifier)

    composite_result_id = str(composite_performance_result)
    uid = composite_performance_result.uid
    output_dir = get_composite_performance_result_output_directory(composite_result_id, uid, suffix=suffix)

    if config:
        log_config(output_dir, config)

    return composite_performance_result, output_dir


def get_composite_performance_result_output_directory(composite_result_id, uid, suffix=None):
    """
    Finds or makes directory for composite performance result outputs. If the directory already exists, verifies that
    the unique id (built using hashes of performance result ids) logged in the directory matches the unique id passed
    (uid). If unique ids match, the directory is returned. Otherwise, the suffix is incremented and the function is
    called again, creating and returning a new directory or returning an existing directory containing a matching
    uid.
    """

    if not suffix:
        output_dir = Path(ROOT_DIR, REL_PATHS['composite_performance'], composite_result_id)
    else:
        dir_name = f'{composite_result_id}-{suffix}'
        output_dir = Path(ROOT_DIR, REL_PATHS['composite_performance'], dir_name)

    if not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)
        log_uid(output_dir, uid)
        return output_dir

    else:
        uid_check = check_cr_uid(output_dir, uid=uid)
        if uid_check is None:  # happens in the event that the directory exists but no uid has been logged yet
            log_uid(output_dir, uid)
            return output_dir

    if uid_check:  # logged uid matches composite result uid
        return output_dir

    else:
        if not suffix:
            suffix = 'v2'
            return get_composite_performance_result_output_directory(composite_result_id, uid, suffix=suffix)
        else:
            suffix = increment_suffix(suffix)
            return get_composite_performance_result_output_directory(composite_result_id, uid, suffix=suffix)


def get_composite_result_uid(performance_prediction_result_ids, eval_result_ids):
    """
    Returns a unique id for each unique set of consisting of performance_prediction_result_ids and eval_result_ids.
    """

    ids = copy.deepcopy(performance_prediction_result_ids)
    ids.sort()

    if eval_result_ids:

        eval_ids = copy.deepcopy(eval_result_ids)
        eval_ids.sort()
        ids.extend(eval_ids)

    uid = blake2b(str(ids).encode('utf-8')).hexdigest()

    return uid


def check_cr_uid(directory, uid=None, composite_result=None):
    """
    Checks to see whether target directory contains matching uid
    """
    if not uid:
        uid = composite_result.uid
    existing_uid = read_existing_uid(directory)

    if existing_uid is None:
        return existing_uid
    else:
        return uid == existing_uid


def read_existing_uid(directory):

    uid_filepath = Path(directory, STANDARD_UID_FILENAME)
    if not uid_filepath.is_file():
        return None

    with open(uid_filepath, 'r') as file:
        uid_log = json.load(file)

    uid = uid_log['uid']
    return uid


def log_uid(directory, uid):

    uid_filepath = Path(directory, STANDARD_UID_FILENAME)
    uid_log = {'uid': uid}

    with open(uid_filepath, 'w') as file:
        json.dump(uid_log, file)


if __name__ == '__main__':

    # config_filename = 's6_oct_fr90_composite_config.yml'
    config_filename = 'pl_oct_composite_fr90_mega1_mega2.yml'

    analyze_1d = False
    analyze_2d = False
    analyze_3d = True

    make_standard_plots = True

    make_simulation_plots_1d = False
    make_simulation_plots_2d = False
    make_isosurf_plots = False

    show_plots = False
    show_1d_plots = True
    show_scatter_plots = False

    print_summary_to_console = True

    fit_keys = [
        'exp_b0n0',
        'exp_b0n1',
        'exp_b0n2',

        'exp_b2n0',
        'exp_b2n1',
        'exp_b2n2',

        'exp_b3n0',
        'exp_b3n1',
        'exp_b3n2',

        'exp_b4n0',
        'exp_b4n1',
        'exp_b4n2',

        'pl_b0n0',  # simplest / naive mapping
        'pl_b0n1',
        'pl_b0n2',

        'pl_b2n0',
        'pl_b2n1',
        'pl_b2n2',

        'pl_b3n0',
        'pl_b3n1',
        'pl_b3n2',  # total noise estimated in quadrature, discrete sampling rer

        'pl_b4n0',
        'pl_b4n1',
        'pl_b4n2',

        'giqe3_b2n0',
        'giqe3_b2n1',
        'giqe3_b2n2',

        'giqe3_b3n0',
        'giqe3_b3n1',
        'giqe3_b3n2',

        'giqe3_b4n0',
        'giqe3_b4n1',
        'giqe3_b4n2',

        'giqe5_b2n0',
        'giqe5_b2n1',
        'giqe5_b2n2',

        'giqe5_b3n0',
        'giqe5_b3n1',
        'giqe5_b3n2',

        'giqe5_b4n0',
        'giqe5_b4n1',
        'giqe5_b4n2',

        #
        # 'giqe3_b2n1',   # no cross-term, noise linearly, pure slope rer, c4 * res squared
        # 'giqe3_b2n2',  # no cross-term, noise in quadrature, pure slope rer, c4 * res squared
        # 'giqe5_b2b2_nct',  # no cross-term, noise in quadrature, pure slope rer, c4 * res squared
        # 'giqe5_b2n2',  # cross-term, noise in quadrature, pure slope rer, c4 * res squared
        # 'giqe5_b3n2_nct',  # no cross-term, noise in quadrature, discrete sampling rer, c4 * res squared
        # 'giqe5_b3n2',  # cross-term, noise in quadrature, discrete sampling rer, c4 * res squared
        # 'giqe3_b3n2',  # no cross-term, noise in quadrature, discrete sampling rer, c4 * res squared
        # 'giqe3_b4n2',  # no cross-term, noise in quadrature, discrete sampling rer/blur corrected, c4 * res squared
        # 'giqe5_b4n2',  # cross-term, noise in quadrature, discrete sampling rer/blur corrected, c4 * res squared
        ]

    performance_fit_summary = {}

    if not config_filename:
        config_filename = 'composite_distortion_analysis_config.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_filename,
                        help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'composite_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _composite_performance, _output_dir = get_composite_performance_result(config=run_config)

    if analyze_1d:
        sub_dir, log_filename = get_sub_dir_and_log_filename(_output_dir, '1d')
        with open(Path(sub_dir, log_filename), 'w') as output_file:
            analyze_perf_1d(_composite_performance, log_file=output_file, directory=sub_dir, per_class=False,
                            distortion_ids=('res', 'blur', 'noise'),
                            show_plots=show_1d_plots)

    if analyze_2d:
        sub_dir, log_filename = get_sub_dir_and_log_filename(_output_dir, '2d')
        with open((Path(sub_dir, log_filename)), 'w') as output_file:
            analyze_perf_2d(_composite_performance, log_file=output_file, directory=sub_dir,
                            distortion_ids=('res', 'blur', 'noise'),
                            show_plots=show_plots)

    if analyze_3d:
        sub_dir_3d, log_filename = get_sub_dir_and_log_filename(_output_dir, '3d')  # got rid of distortion_clip
        with open((Path(sub_dir_3d, log_filename)), 'w') as output_file:
            for _fit_key in fit_keys:
                fit_sub_dir, ___ = get_sub_dir_and_log_filename(sub_dir_3d, _fit_key)
                fit_summary_stats = analyze_perf_3d(_composite_performance, log_file=output_file, directory=fit_sub_dir,
                                                    fit_key=_fit_key, standard_plots=make_standard_plots,
                                                    residual_plot=False, make_residual_color_plot=False,
                                                    distortion_ids=('res', 'blur', 'noise'),
                                                    isosurf_plot=make_isosurf_plots,
                                                    make_simulation_plots_1d=make_simulation_plots_1d,
                                                    make_simulation_plots_2d=make_simulation_plots_2d,
                                                    show_plots=show_plots,
                                                    show_1d_plots=show_1d_plots,
                                                    show_scatter_plots=show_scatter_plots)
                performance_fit_summary[_fit_key] = fit_summary_stats

            performance_fit_summary_text_dump(performance_fit_summary, file=output_file,
                                              print_to_console=print_summary_to_console)
