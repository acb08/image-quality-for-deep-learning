from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, STANDARD_TEST_RESULT_FILENAME, PROJECT_ID, REL_PATHS, \
    ROOT_DIR
from src.d00_utils.functions import load_wandb_data_artifact, get_config
from src.d04_analysis.performance_analysis_functions import conditional_mean_accuracy, \
    extract_combine_shard_vector_data, extract_embedded_vectors
from src.d04_analysis.fit import fit_hyperplane, eval_linear_fit, linear_predict
from src.d04_analysis.tools3d import conditional_extract_2d, wire_plot, build_3d_field, plot_isosurf
from src.d04_analysis.plot_defaults import AZ_EL_DEFAULTS, AXIS_LABELS, AZ_EL_COMBINATIONS, COLORS, SCATTER_PLOT_MARKERS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import wandb
import argparse

wandb.login()


# class DistortedDataset(object):
#
#     def __init__(self, dataset, convert_to_std=True):
#         self.dataset = dataset
#         self.res, self.blur, self.noise = extract_distortion_vectors(self.dataset)
#         self.res = np.asarray(self.res)
#         self.blur = np.asarray(self.blur)
#         self.noise = np.asarray(self.noise)
#         if convert_to_std:
#             self.noise = np.sqrt(self.noise)
#         self.distortions = {
#             'res': self.res,
#             'blur': self.blur,
#             'noise': self.noise
#         }


class DistortedDataset(object):

    def __init__(self, dataset,
                 intermediate_keys=('test', 'image_distortion_info'),
                 target_keys=('res', 'blur', 'noise'),
                 convert_to_std=True):
        self.dataset = dataset
        self.res, self.blur, self.noise = extract_embedded_vectors(self.dataset,
                                                                   intermediate_keys=intermediate_keys,
                                                                   target_keys=target_keys,
                                                                   return_full_dict=False)
        self.res = np.asarray(self.res)
        self.blur = np.asarray(self.blur)
        self.noise = np.asarray(self.noise)
        if convert_to_std:
            self.noise = np.sqrt(self.noise)
        self.distortions = {
            'res': self.res,
            'blur': self.blur,
            'noise': self.noise
        }


# class ModelDistortionPerformanceResult(DistortedDataset):
#
#     def __init__(self, run, result_id, identifier=None, convert_to_std=True):
#         self.convert_to_std = convert_to_std
#         self.dataset, self.result = load_dataset_and_result(run, result_id)
#         DistortedDataset.__init__(self, self.dataset, convert_to_std=self.convert_to_std)
#         self.labels, self.predicts = extract_performance_vectors(self.result)
#         self.predicts = np.asarray(self.predicts)
#         self.labels = np.asarray(self.labels)
#         self.top_1_vec = self.get_accuracy_vector()
#         self.identifier = identifier
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __repr__(self):
#         return str(self.identifier)
#
#     def get_accuracy_vector(self):
#         return np.equal(self.labels, self.predicts)
#
#     def conditional_accuracy(self, distortion_id):
#         return conditional_mean_accuracy(self.labels, self.predicts, self.distortions[distortion_id])


class ModelDistortionPerformanceResult(DistortedDataset):

    def __init__(self, run, result_id, identifier=None, convert_to_std=True):
        self.convert_to_std = convert_to_std
        self.dataset, self.result = load_dataset_and_result(run, result_id)
        DistortedDataset.__init__(self, self.dataset, convert_to_std=self.convert_to_std)
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

    def conditional_accuracy(self, distortion_id):
        return conditional_mean_accuracy(self.labels, self.predicts, self.distortions[distortion_id])


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

    return dataset, result


# def extract_distortion_vectors(dataset,
#                                dataset_split_key='test',
#                                distortion_info_key='image_distortion_info',
#                                distortion_type_flags=('res', 'blur', 'noise'),
#                                return_full_dict=False):
#     image_distortion_info = dataset[dataset_split_key][distortion_info_key]
#     extracted_distortion_data = extract_combine_shard_vector_data(image_distortion_info, distortion_type_flags)
#
#     if return_full_dict:
#         return extracted_distortion_data
#
#     else:
#         distortion_vectors = []
#         for flag in distortion_type_flags:
#             distortion_vectors.append(extracted_distortion_data[flag])
#
#         return distortion_vectors


# def extract_performance_vectors(test_result,
#                                 performance_key='shard_performances',
#                                 target_vector_keys=('labels', 'predicts'),
#                                 return_full_dict=False):
#     performance_data = test_result[performance_key]
#     extracted_performance_vectors = extract_combine_shard_vector_data(performance_data, target_vector_keys)
#
#     if return_full_dict:
#         return extracted_performance_vectors
#
#     else:
#         performance_vectors = []
#         for target_vector_key in target_vector_keys:
#             performance_vectors.append(extracted_performance_vectors[target_vector_key])
#
#         return performance_vectors
#

def plot_1d_linear_fit(x_data, y_data, fit_coefficients, distortion_id,
                       result_identifier=None, ylabel='accuracy', title=None, directory=None):
    xlabel = AXIS_LABELS[distortion_id]
    x_plot = np.linspace(np.min(x_data), np.max(x_data), num=50)
    y_plot = fit_coefficients[0] * x_plot + fit_coefficients[1]

    ax = plt.figure().gca()

    ax.plot(x_plot, y_plot, linestyle='dashed', lw=0.8, color='k')
    ax.scatter(x_data, y_data)
    ax.set_xlabel(xlabel)
    if 'noise' in xlabel or np.max(x_data) > 5:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if directory:
        if result_identifier:
            save_name = f'{distortion_id}_{result_identifier}_{ylabel}.png'
        else:
            save_name = f'{distortion_id}_{ylabel}.png'
        plt.savefig(Path(directory, save_name))
    plt.show()


def get_distortion_perf_1d(model_performance, distortion_id, log_file=None, add_bias=True):
    result_name = str(model_performance)
    distortion_vals, mean_accuracies = model_performance.conditional_accuracy(distortion_id)

    fit_coefficients = fit_hyperplane(np.atleast_2d(distortion_vals).T,
                                      np.atleast_2d(mean_accuracies).T,
                                      add_bias=add_bias)

    correlation = eval_linear_fit(fit_coefficients,
                                  np.atleast_2d(distortion_vals).T,
                                  np.atleast_2d(mean_accuracies).T)

    print(f'{result_name} {distortion_id} linear fit: ', fit_coefficients, file=log_file)
    print(f'{result_name} {distortion_id} linear fit correlation: ', correlation, '\n', file=log_file)

    return distortion_vals, mean_accuracies, fit_coefficients, correlation


def plot_2d_performance(x_values, y_values, accuracy_means, x_id, y_id,
                        result_identifier=None,
                        axis_labels=None,
                        az_el_combinations='all',
                        directory=None):
    if not axis_labels or axis_labels == 'default':
        xlabel, ylabel = AXIS_LABELS[x_id], AXIS_LABELS[y_id]
    else:
        xlabel, ylabel = axis_labels[x_id], axis_labels[x_id]

    if result_identifier:
        save_name = f'{x_id}_{y_id}_{str(result_identifier)}_acc.png'
    else:
        save_name = f'{x_id}_{y_id}_acc.png'

    if az_el_combinations == 'all':

        for combination_key in AZ_EL_COMBINATIONS:
            az, el = AZ_EL_COMBINATIONS[combination_key]['az'], AZ_EL_COMBINATIONS[combination_key]['el']

            wire_plot(x_values, y_values, accuracy_means,
                      xlabel=xlabel, ylabel=ylabel,
                      az=az, el=el,
                      save_name=save_name,
                      directory=directory)

    else:
        if az_el_combinations == 'default':
            az, el = AZ_EL_DEFAULTS['az'], AZ_EL_DEFAULTS['el']
        elif az_el_combinations in AZ_EL_COMBINATIONS:
            az, el = AZ_EL_COMBINATIONS[az_el_combinations]['az'], AZ_EL_COMBINATIONS[az_el_combinations]['el']
        else:
            az, el = az_el_combinations[0], az_el_combinations[1]

        wire_plot(x_values, y_values, accuracy_means,
                  xlabel=xlabel, ylabel=ylabel,
                  az=az, el=el,
                  save_name=save_name,
                  directory=directory)


def plot_2d_linear_fit(distortion_array, accuracy_means, fit, x_id, y_id,
                       result_identifier=None, axis_labels='default', az_el_combinations='all', directory=None):
    x, y = distortion_array[:, 0], distortion_array[:, 1]
    predict_mean_accuracy_vector = linear_predict(fit, distortion_array)
    x_values, y_values, predicted_mean_accuracy, __ = conditional_extract_2d(x, y, predict_mean_accuracy_vector)

    z_plot = {
        'fit': predicted_mean_accuracy,
        'actual': accuracy_means
    }

    plot_2d_performance(x_values, y_values, z_plot, x_id, y_id,
                        axis_labels=axis_labels, az_el_combinations=az_el_combinations, directory=directory,
                        result_identifier=result_identifier)


def analyze_perf_1d(model_performance,
                    distortion_ids=('res', 'blur', 'noise'),
                    directory=None,
                    log_file=None):
    for i, distortion_id in enumerate(distortion_ids):
        x, y, fit_coefficients, fit_correlation = get_distortion_perf_1d(model_performance, distortion_id,
                                                                         log_file=log_file)
        plot_1d_linear_fit(x, y, fit_coefficients, distortion_id,
                           result_identifier=str(model_performance), directory=directory)


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

        plot_2d_performance(x_values, y_values, accuracy_means, x_id, y_id,
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

        plot_2d_performance(x_values, y_values, mean_performances, x_id, y_id,
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
                        legend_loc='best'):

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
    correlation = eval_linear_fit(fit_coefficients, distortion_array, perf_array, add_bias=True)

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


def get_model_distortion_performance_result(result_id=None, identifier=None, config=None):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    with wandb.init(project=PROJECT_ID, job_type='analyze_test_result') as run:
        output_dir = Path(ROOT_DIR, REL_PATHS['analysis'], result_id)
        if not output_dir.is_dir():
            Path.mkdir(output_dir)

        model_distortion_performance = ModelDistortionPerformanceResult(run, result_id, identifier=identifier)

    return model_distortion_performance, output_dir


# def get_model_distortion_performance_result_compare(result_id=None, identifier=None, config=None):
#
#     if not result_id and not identifier:
#         result_id = config['result_id']
#         identifier = config['identifier']
#
#     with wandb.init(project=PROJECT_ID, job_type='analyze_test_result') as run:
#         output_dir = Path(ROOT_DIR, REL_PATHS['analysis'], result_id)
#         if not output_dir.is_dir():
#             Path.mkdir(output_dir)
#
#         model_distortion_performance = ModelDistortionPerformanceResultV2(run, result_id, identifier=identifier)
#
#     return model_distortion_performance, output_dir


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
        analyze_perf_1d(_model_distortion_performance, log_file=output_file, directory=_output_dir)
        analyze_perf_2d(_model_distortion_performance, log_file=output_file, directory=_output_dir)
        analyze_perf_3d(_model_distortion_performance, log_file=output_file, directory=_output_dir)