import copy
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, STANDARD_TEST_RESULT_FILENAME, PROJECT_ID, REL_PATHS, \
    ROOT_DIR
from src.d00_utils.functions import load_wandb_data_artifact, get_config, construct_artifact_id
from src.d04_analysis._shared_methods import _get_processed_instance_props_path, _check_extract_processed_props, \
    _archive_processed_props, _get_3d_distortion_perf_props
from src.d04_analysis.analysis_functions import conditional_mean_accuracy, extract_embedded_vectors, \
    get_class_accuracies, build_3d_field, get_distortion_perf_2d, get_distortion_perf_1d
from src.d04_analysis.fit import fit_hyperplane, eval_linear_fit
from src.d04_analysis.plot import plot_1d_linear_fit, plot_2d, plot_2d_linear_fit, plot_isosurf
import numpy as np
from pathlib import Path
import wandb
import argparse

wandb.login()


class DistortedDataset(object):

    def __init__(self, dataset,
                 intermediate_keys=('test', 'image_distortion_info'),
                 distortion_ids=('res', 'blur', 'noise'),
                 convert_to_std=True):
        self._dataset = dataset
        self.distortion_ids = distortion_ids
        self._distortion_data = extract_embedded_vectors(self._dataset,
                                                         intermediate_keys=intermediate_keys,
                                                         target_keys=self.distortion_ids,
                                                         return_full_dict=False)

        self.res = None
        self.blur = None
        self.noise = None
        self.scaled_blur = None

        for i, distortion_id in enumerate(self.distortion_ids):
            distortion = self._distortion_data[i]
            if distortion_id == 'res':
                self.res = distortion
            elif distortion_id == 'blur':
                self.blur = distortion
            elif distortion_id == 'noise':
                self.noise = distortion

        if self.res is not None:
            self.res = np.asarray(self.res)
        if self.blur is not None:
            self.blur = np.asarray(self.blur)
        if self.noise is not None:
            self.noise = np.asarray(self.noise)

        if self.res is not None and self.blur is not None:
            self.scaled_blur = self.blur / self.res
            self.scaled_blur = self._quantize_scaled_blur()

        if self.noise is not None and convert_to_std:
            self.noise = np.sqrt(self.noise)
        self.distortions = {
            'res': self.res,
            'blur': self.blur,
            'scaled_blur': self.scaled_blur,
            'noise': self.noise
        }

    def _quantize_scaled_blur(self):
        blur_vals = np.unique(self.blur)
        quantized_scaled_blur_vals = blur_vals
        while max(quantized_scaled_blur_vals) < np.max(self.scaled_blur):
            extension = blur_vals + max(quantized_scaled_blur_vals)
            quantized_scaled_blur_vals = np.append(quantized_scaled_blur_vals, extension)
        quantization_indices = np.digitize(self.scaled_blur, quantized_scaled_blur_vals, right=True)
        quantized_scaled_blur = quantized_scaled_blur_vals[quantization_indices]
        return quantized_scaled_blur


class ModelDistortionPerformanceResult(DistortedDataset):

    def __init__(self, run, result_id, identifier=None, convert_to_std=True, distortion_ids=('res', 'blur', 'noise')):
        self.convert_to_std = convert_to_std
        self.result_id = result_id
        self._dataset, self._result, self.dataset_id = load_dataset_and_result(run, self.result_id)
        self._model_artifact_id = self._result['model_artifact_id']
        self._model_artifact_alias = self._result['model_artifact_alias']
        self.model_id = f'{self._model_artifact_id}:{self._model_artifact_alias}'
        self.distortion_ids = distortion_ids
        DistortedDataset.__init__(self, self._dataset, convert_to_std=self.convert_to_std,
                                  distortion_ids=self.distortion_ids)
        self.labels, self.predicts = extract_embedded_vectors(self._result,
                                                              intermediate_keys=['shard_performances'],
                                                              target_keys=('labels', 'predicts'),
                                                              return_full_dict=False)
        self.predicts = np.asarray(self.predicts)
        self.labels = np.asarray(self.labels)
        self.top_1_vec = self.get_accuracy_vector()
        self.identifier = identifier
        self.instance_hash = hash(tuple(self.top_1_vec))

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        if self.identifier:
            return str(self.identifier)
        else:
            return self.__repr__()

    def __repr__(self):
        return self.result_id

    def get_accuracy_vector(self):
        return np.equal(self.labels, self.predicts)

    def conditional_accuracy(self, distortion_id, per_class=False):
        return conditional_mean_accuracy(self.labels, self.predicts, self.distortions[distortion_id],
                                         per_class=per_class)

    def class_accuracies(self):
        classes, class_accuracies = get_class_accuracies(self.labels, self.predicts)
        return classes, class_accuracies

    def mean_per_class_accuracy(self):
        __, class_accuracies = self.class_accuracies()
        return np.mean(class_accuracies)

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


def load_dataset_and_result(run, result_id,
                            result_alias='latest',
                            test_dataset_id_key='test_dataset_id',
                            test_dataset_alias_key='test_dataset_artifact_alias',
                            dataset_filename=STANDARD_DATASET_FILENAME,
                            result_filename=STANDARD_TEST_RESULT_FILENAME):
    if ':' not in result_id:
        result_id = f'{result_id}:{result_alias}'

    result_dir, result = load_wandb_data_artifact(run, result_id, result_filename)

    dataset_id = result[test_dataset_id_key]
    dataset_artifact_alias = result[test_dataset_alias_key]
    dataset_id = f'{dataset_id}:{dataset_artifact_alias}'
    dataset_dir, dataset = load_wandb_data_artifact(run, dataset_id, dataset_filename)

    return dataset, result, dataset_id


def analyze_perf_1d(model_performance,
                    distortion_ids=('res', 'blur', 'noise'),
                    directory=None,
                    log_file=None,
                    per_class=False):
    for i, distortion_id in enumerate(distortion_ids):
        x, y, fit_coefficients, fit_correlation = get_distortion_perf_1d(model_performance, distortion_id,
                                                                         log_file=log_file, per_class=per_class)
        plot_1d_linear_fit(x, y, fit_coefficients, distortion_id,
                           result_identifier=str(model_performance), directory=directory, per_class=per_class)


def analyze_perf_2d(model_performance,
                    distortion_ids=('res', 'blur', 'noise'),
                    distortion_combinations=((0, 1), (1, 2), (0, 2)),
                    directory=None,
                    log_file=None,
                    add_bias=True):
    identifier = str(model_performance)

    for i, (idx_0, idx_1) in enumerate(distortion_combinations):
        x_id, y_id = distortion_ids[idx_0], distortion_ids[idx_1]

        x_values, y_values, accuracy_means, fit, corr, distortion_arr = get_distortion_perf_2d(model_performance,
                                                                                               x_id,
                                                                                               y_id,
                                                                                               add_bias=add_bias,
                                                                                               log_file=log_file)

        plot_2d(x_values, y_values, accuracy_means, x_id, y_id,
                result_identifier=identifier,
                axis_labels='default',
                az_el_combinations='all',
                directory=directory)

        plot_2d_linear_fit(distortion_arr, accuracy_means, fit, x_id, y_id,
                           result_identifier=f'{identifier}_fit',
                           axis_labels='default',
                           az_el_combinations='all',
                           directory=directory)


def get_distortion_perf_3d(model_performance, x_id='res', y_id='blur', z_id='noise', add_bias=True, log_file=None):
    result_name = str(model_performance)

    try:
        x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = (
            model_performance.get_3d_distortion_perf_props(distortion_ids=(x_id, y_id, z_id)))
    except ValueError:  # raised by get_3d_distortion_perf_props if distortion_ids != ('res', 'blur', 'noise')

        x = model_performance.distortions[x_id]
        y = model_performance.distortions[y_id]
        z = model_performance.distortions[z_id]
        accuracy_vector = model_performance.perf_predict_top_1_array

        x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(x, y, z,
                                                                                                 accuracy_vector,
                                                                                                 data_dump=True)

    fit_coefficients = fit_hyperplane(distortion_array, perf_array, add_bias=add_bias)
    correlation = eval_linear_fit(fit_coefficients, distortion_array, perf_array, add_bias=add_bias)

    print(f'{result_name} {x_id} {y_id} {z_id} linear fit: ', fit_coefficients, file=log_file)
    print(f'{result_name} {x_id} {y_id} {z_id} linear fit correlation: ', correlation, '\n', file=log_file)

    return x_values, y_values, z_values, perf_3d


def analyze_perf_3d(model_performance,
                    distortion_ids=('res', 'blur', 'noise'),
                    log_file=None,
                    add_bias=True,
                    directory=None,
                    isosurf_plot=False):
    x_id, y_id, z_id = distortion_ids
    x_values, y_values, z_values, perf_3d = get_distortion_perf_3d(model_performance,
                                                                   x_id=x_id, y_id=y_id, z_id=z_id,
                                                                   add_bias=add_bias, log_file=log_file)
    if isosurf_plot:
        save_name = f'{str(model_performance)}_isosurf.png'
        plot_isosurf(x_values, y_values, z_values, perf_3d,
                     level=np.mean(perf_3d), save_name=save_name, save_dir=directory)


def check_extraction_method(model_performance):

    built_in = model_performance.get_3d_distortion_perf_props(('res', 'blur', 'noise'))
    built_in_compare = built_in[:4]
    original = get_distortion_perf_3d(model_performance)

    for i, arr in enumerate(original):
        print(np.array_equal(arr, built_in_compare[i]))


def _fetch_model_distortion_performance_result(run, result_id, identifier, distortion_ids, make_dir=True):

    output_dir = Path(ROOT_DIR, REL_PATHS['analysis'], result_id)
    if make_dir and not output_dir.is_dir():
        Path.mkdir(output_dir)

    model_distortion_performance = ModelDistortionPerformanceResult(run, result_id, identifier=identifier,
                                                                    distortion_ids=distortion_ids)

    return model_distortion_performance, output_dir


def get_model_distortion_performance_result(result_id=None, identifier=None, config=None,
                                            distortion_ids=('res', 'blur', 'noise'), make_dir=True):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    with wandb.init(project=PROJECT_ID, job_type='analyze_test_result') as run:
        model_distortion_performance, output_dir = _fetch_model_distortion_performance_result(run, result_id,
                                                                                              identifier,
                                                                                              distortion_ids,
                                                                                              make_dir=make_dir)
    return model_distortion_performance, output_dir


def get_multiple_model_distortion_performance_results(result_id_pairs, distortion_ids=('res', 'blur', 'noise'),
                                                      make_dir=True, output_type='list'):

    if output_type == 'list':
        performance_results = []
    elif output_type == 'dict':
        performance_results = {}
    else:
        raise Exception('invalid output_type')

    if type(result_id_pairs[0]) == str:  # create tuples with dummy identifiers for if a simple list passed
        result_id_pairs = copy.deepcopy(result_id_pairs)
        result_id_pairs = [(result_id, i) for i, result_id in enumerate(result_id_pairs)]

    with wandb.init(project=PROJECT_ID, job_type='analyze_test_result') as run:

        for (artifact_id, identifier) in result_id_pairs:

            artifact_id, __ = construct_artifact_id(artifact_id)

            performance_result, __ = _fetch_model_distortion_performance_result(run, artifact_id, identifier,
                                                                                distortion_ids, make_dir=make_dir)
            if output_type == 'list':
                performance_results.append(performance_result)
            elif output_type == 'dict':
                performance_results[artifact_id] = performance_result

    return performance_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_analysis_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_analysis_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _model_distortion_performance, _output_dir = get_model_distortion_performance_result(config=run_config)

    with open(Path(_output_dir, 'result_log.txt'), 'w') as output_file:
        analyze_perf_1d(_model_distortion_performance, log_file=output_file, directory=_output_dir, per_class=False,
                        distortion_ids=('res', 'blur', 'scaled_blur', 'noise'))
        analyze_perf_1d(_model_distortion_performance, log_file=output_file, directory=_output_dir, per_class=True,
                        distortion_ids=('res', 'blur', 'scaled_blur', 'noise'))
        analyze_perf_2d(_model_distortion_performance, log_file=output_file, directory=_output_dir)
        analyze_perf_2d(_model_distortion_performance, log_file=output_file, directory=_output_dir,
                        distortion_ids=('scaled_blur', 'noise'), distortion_combinations=((0, 1),))
        analyze_perf_3d(_model_distortion_performance, log_file=output_file, directory=_output_dir)
