from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, STANDARD_TEST_RESULT_FILENAME, PROJECT_ID, REL_PATHS, \
    ROOT_DIR
from src.d00_utils.functions import load_wandb_data_artifact, get_config
from src.d04_analysis.analysis_functions import conditional_mean_accuracy, extract_embedded_vectors, \
    conditional_extract_2d, get_class_accuracies
from src.d04_analysis.fit import fit_hyperplane, eval_linear_fit
from src.d04_analysis.tools3d import build_3d_field, plot_isosurf
from src.d04_analysis.plot import AXIS_LABELS, COLORS, SCATTER_PLOT_MARKERS, \
    plot_1d_linear_fit, plot_2d, plot_2d_linear_fit
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
import argparse

wandb.login()


class DistortedDataset(object):

    def __init__(self, dataset,
                 intermediate_keys=('test', 'image_distortion_info'),
                 distortion_ids=('res', 'blur', 'noise'),
                 convert_to_std=True):
        self.dataset = dataset
        self.distortion_ids = distortion_ids
        # self.res, self.blur, self.noise = extract_embedded_vectors(self.dataset,
        #                                                            intermediate_keys=intermediate_keys,
        #                                                            target_keys=self.distortion_ids,
        #                                                            return_full_dict=False)
        self._distortion_data = extract_embedded_vectors(self.dataset,
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
            self.scaled_blur = self.quantize_scaled_blur()

        if self.noise is not None and convert_to_std:
            self.noise = np.sqrt(self.noise)
        self.distortions = {
            'res': self.res,
            'blur': self.blur,
            'scaled_blur': self.scaled_blur,
            'noise': self.noise
        }

    def quantize_scaled_blur(self):
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
        self.dataset, self.result, self.dataset_id = load_dataset_and_result(run, self.result_id)
        self.distortion_ids = distortion_ids
        DistortedDataset.__init__(self, self.dataset, convert_to_std=self.convert_to_std,
                                  distortion_ids=self.distortion_ids)
        self.labels, self.predicts = extract_embedded_vectors(self.result,
                                                              intermediate_keys=['shard_performances'],
                                                              target_keys=('labels', 'predicts'),
                                                              return_full_dict=False)
        self.predicts = np.asarray(self.predicts)
        self.labels = np.asarray(self.labels)
        self.top_1_vec = self.get_accuracy_vector()
        self.identifier = identifier

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return str(self.identifier)

    def get_accuracy_vector(self):
        return np.equal(self.labels, self.predicts)

    def conditional_accuracy(self, distortion_id, per_class=False):
        # if per_class==True, returns mean per class accuracy rather than overall mean accuracy
        return conditional_mean_accuracy(self.labels, self.predicts, self.distortions[distortion_id],
                                         per_class=per_class)

    def class_accuracies(self):
        classes, class_accuracies = get_class_accuracies(self.labels, self.predicts)
        return classes, class_accuracies

    def mean_per_class_accuracy(self):
        __, class_accuracies = self.class_accuracies()
        return np.mean(class_accuracies)


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


def get_distortion_perf_1d(model_performance, distortion_id, log_file=None, add_bias=True, per_class=False):
    result_name = str(model_performance)
    distortion_vals, mean_accuracies = model_performance.conditional_accuracy(distortion_id, per_class=per_class)

    fit_coefficients = fit_hyperplane(np.atleast_2d(distortion_vals).T,
                                      np.atleast_2d(mean_accuracies).T,
                                      add_bias=add_bias)

    correlation = eval_linear_fit(fit_coefficients,
                                  np.atleast_2d(distortion_vals).T,
                                  np.atleast_2d(mean_accuracies).T)

    print(f'{result_name} {distortion_id} (per_class = {per_class}) linear fit: ', fit_coefficients, file=log_file)
    print(f'{result_name} {distortion_id} (per_class = {per_class}) linear fit correlation: ', correlation, '\n', file=log_file)

    return distortion_vals, mean_accuracies, fit_coefficients, correlation


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


def plot_perf_2d_multi_result(model_performances,
                              distortion_ids=('res', 'blur', 'noise'),
                              distortion_combinations=((0, 1), (1, 2), (0, 2)),
                              directory=None,
                              log_file=None,
                              add_bias=True,
                              identifier=None):

    for i, (idx_0, idx_1) in enumerate(distortion_combinations):

        x_id, y_id = distortion_ids[idx_0], distortion_ids[idx_1]
        mean_performances = {}

        for model_performance in model_performances:
            performance_key = str(model_performance)
            x_values, y_values, accuracy_means, fit, corr, distortion_arr = get_distortion_perf_2d(model_performance,
                                                                                                   x_id,
                                                                                                   y_id,
                                                                                                   add_bias=add_bias,
                                                                                                   log_file=log_file)
            mean_performances[performance_key] = accuracy_means

        plot_2d(x_values, y_values, mean_performances, x_id, y_id,
                result_identifier=identifier,
                axis_labels='default',
                az_el_combinations='all',
                directory=directory)


def plot_perf_1d_multi_result(model_performances,
                              distortion_ids=('res', 'blur', 'noise'),
                              directory=None,
                              identifier=None,
                              legend_loc='best'):
    """
    :param model_performances: list of model performance class instances
    :param distortion_ids: distortion type tags to be analyzed
    :param directory: output derectory
    :param identifier: str for use as a filename seed
    :param legend_loc: str to specify plot legend location
    """

    for i, distortion_id in enumerate(distortion_ids):
        mean_performances = {}
        for model_performance in model_performances:
            performance_key = str(model_performance)
            x, y, fit_coefficients, fit_correlation = get_distortion_perf_1d(model_performance, distortion_id)
            mean_performances[performance_key] = y

        plot_1d_performance(x, mean_performances, distortion_id, result_identifier=identifier, directory=directory,
                            legend_loc=legend_loc)


def plot_1d_performance(x, performance_dict, distortion_id,
                        result_identifier=None,
                        xlabel='default',
                        ylabel='default',
                        directory=None,
                        legend_loc='best',
                        legend=True):

    if xlabel == 'default':
        xlabel = AXIS_LABELS[distortion_id]
    if ylabel == 'default':
        ylabel = AXIS_LABELS['y']

    if result_identifier:
        save_name = f'{distortion_id}_{str(result_identifier)}_acc'
    else:
        save_name = f'{distortion_id}_acc'

    if legend_loc and legend_loc != 'best':
        save_name = f"{save_name}_{legend_loc.replace(' ', '_')}"

    save_name = f"{save_name}.png"

    plt.figure()
    for i, key in enumerate(performance_dict):
        plt.plot(x, performance_dict[key], color=COLORS[i])
        plt.scatter(x, performance_dict[key], label=key, c=COLORS[i], marker=SCATTER_PLOT_MARKERS[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(loc=legend_loc)
    if directory:
        plt.savefig(Path(directory, save_name))
    plt.show()


def get_distortion_perf_2d(model_performance, x_id, y_id, add_bias=True, log_file=None):
    result_name = str(model_performance)

    accuracy_vector = model_performance.top_1_vec
    x = model_performance.distortions[x_id]
    y = model_performance.distortions[y_id]
    x_values, y_values, accuracy_means, vector_data_extract = conditional_extract_2d(x, y, accuracy_vector)

    distortion_param_array = vector_data_extract['param_array']
    performance_array = vector_data_extract['performance_array']
    fit_coefficients = fit_hyperplane(distortion_param_array, performance_array, add_bias=add_bias)
    correlation = eval_linear_fit(fit_coefficients, distortion_param_array, performance_array, add_bias=add_bias)

    print(f'{result_name} {x_id} {y_id} linear fit: ', fit_coefficients, file=log_file)
    print(f'{result_name} {x_id} {y_id} linear fit correlation: ', correlation, '\n', file=log_file)

    return x_values, y_values, accuracy_means, fit_coefficients, correlation, distortion_param_array


def get_distortion_perf_3d(model_performance, x_id='res', y_id='blur', z_id='noise', add_bias=True, log_file=None):
    result_name = str(model_performance)

    x = model_performance.distortions[x_id]
    y = model_performance.distortions[y_id]
    z = model_performance.distortions[z_id]
    accuracy_vector = model_performance.top_1_vec

    x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(x, y, z, accuracy_vector,
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


def get_model_distortion_performance_result(result_id=None, identifier=None, config=None,
                                            distortion_ids=('res', 'blur', 'noise'), make_dir=True):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    with wandb.init(project=PROJECT_ID, job_type='analyze_test_result') as run:
        output_dir = Path(ROOT_DIR, REL_PATHS['analysis'], result_id)
        if make_dir and not output_dir.is_dir():
            Path.mkdir(output_dir)

        model_distortion_performance = ModelDistortionPerformanceResult(run, result_id, identifier=identifier,
                                                                        distortion_ids=distortion_ids)

    return model_distortion_performance, output_dir


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
