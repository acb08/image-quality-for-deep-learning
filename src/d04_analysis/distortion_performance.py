import copy
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, STANDARD_TEST_RESULT_FILENAME, WANDB_PID, REL_PATHS, \
    ROOT_DIR, STANDARD_PERFORMANCE_PREDICTION_FILENAME, STANDARD_FIT_STATS_FILENAME
from src.d00_utils.functions import load_wandb_data_artifact, get_config, construct_artifact_id
from src.d04_analysis._shared_methods import _get_processed_instance_props_path, _check_extract_processed_props, \
    _archive_processed_props, _get_3d_distortion_perf_props
from src.d04_analysis.analysis_functions import conditional_mean_accuracy, extract_embedded_vectors, \
    get_class_accuracies, build_3d_field, get_distortion_perf_2d, get_distortion_perf_1d, check_durbin_watson_statistics
from src.d04_analysis.fit import fit, evaluate_fit, apply_fit
from src.d04_analysis.plot import plot_1d_linear_fit, plot_2d, plot_2d_linear_fit, plot_isosurf, compare_2d_views, \
    residual_color_plot, sorted_linear_scatter, compare_1d_views, dual_sorted_linear_scatter
from src.d04_analysis.binomial_simulation import get_ideal_correlation, run_binomial_accuracy_experiment
import numpy as np
from pathlib import Path
import wandb
import argparse
import matplotlib.pyplot as plt
from hashlib import blake2b
from yaml import dump


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
        self.top_1_vec_predict = self.top_1_vec  # needed for forward compatibility where CompositePerformanceResult
        # objects are also used
        self.identifier = identifier
        self.instance_hashes = {'predict': blake2b(str(self.top_1_vec).encode('utf-8')).hexdigest()}

        # _predict attributes used for forward compatibility with scripts using CompositePerformanceResult class
        self.res_predict = self.res
        self.blur_predict = self.blur
        self.noise_predict = self.noise

        # allows access of the form model_performance.distortions[predict_eval_flag][distortion_id]
        self.distortions['predict'] = copy.deepcopy(self.distortions)
        self.distortions['eval'] = copy.deepcopy(self.distortions)

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

    def mean_accuracy(self):
        return np.mean(self.top_1_vec)

    def conditional_accuracy(self, distortion_id, per_class=False):
        return conditional_mean_accuracy(self.labels, self.predicts, self.distortions[distortion_id],
                                         per_class=per_class)

    def class_accuracies(self):
        classes, class_accuracies = get_class_accuracies(self.labels, self.predicts)
        return classes, class_accuracies

    def mean_per_class_accuracy(self):
        __, class_accuracies = self.class_accuracies()
        return np.mean(class_accuracies)

    def get_processed_instance_props_path(self, predict_eval_flag='predict'):
        return _get_processed_instance_props_path(self, predict_eval_flag=predict_eval_flag)

    def check_extract_processed_props(self, predict_eval_flag=None):
        return _check_extract_processed_props(self, predict_eval_flag=predict_eval_flag)

    def archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                perf_array, predict_eval_flag):
        return _archive_processed_props(self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                                        perf_array, predict_eval_flag)

    def get_3d_distortion_perf_props(self, distortion_ids, predict_eval_flag='predict'):
        return _get_3d_distortion_perf_props(self, distortion_ids, predict_eval_flag=predict_eval_flag)


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
                    per_class=False,
                    show_plots=False):
    for i, distortion_id in enumerate(distortion_ids):
        x, y, fit_coefficients, fit_correlation = get_distortion_perf_1d(model_performance, distortion_id,
                                                                         log_file=log_file, per_class=per_class)
        plot_1d_linear_fit(x, y, fit_coefficients, distortion_id,
                           result_identifier=str(model_performance), directory=directory, per_class=per_class,
                           show_plots=show_plots)


def analyze_perf_2d(model_performance,
                    distortion_ids=('res', 'blur', 'noise'),
                    distortion_combinations=((0, 1), (1, 2), (0, 2)),
                    directory=None,
                    log_file=None,
                    add_bias=True,
                    show_plots=False):
    identifier = str(model_performance)

    for i, (idx_0, idx_1) in enumerate(distortion_combinations):
        x_id, y_id = distortion_ids[idx_0], distortion_ids[idx_1]

        x_values, y_values, accuracy_means, _fit, corr, distortion_arr = get_distortion_perf_2d(model_performance,
                                                                                                x_id,
                                                                                                y_id,
                                                                                                add_bias=add_bias,
                                                                                                log_file=log_file)

        plot_2d(x_values, y_values, accuracy_means, x_id, y_id,
                result_identifier=identifier,
                axis_labels='default',
                az_el_combinations='all',
                directory=directory,
                show_plots=show_plots)

        plot_2d_linear_fit(distortion_arr, accuracy_means, _fit, x_id, y_id,
                           result_identifier=f'{identifier}_fit',
                           axis_labels='default',
                           az_el_combinations='all',
                           directory=directory,
                           show_plots=show_plots)


def get_distortion_perf_3d(model_performance, x_id='res', y_id='blur', z_id='noise', add_bias=True, log_file=None,
                           fit_key='linear',):

    result_name = str(model_performance)

    try:
        x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = (
            model_performance.get_3d_distortion_perf_props(distortion_ids=(x_id, y_id, z_id),
                                                           predict_eval_flag='predict'))
    except ValueError:  # raised by get_3d_distortion_perf_props if distortion_ids != ('res', 'blur', 'noise')
        accuracy_vector = model_performance.top_1_vec_predict
        x = model_performance.distortions['predict'][x_id]
        y = model_performance.distortions['predict'][y_id]
        z = model_performance.distortions['predict'][z_id]

        x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(x, y, z,
                                                                                                 accuracy_vector,
                                                                                                 data_dump=True)

    fit_coefficients = fit(distortion_array, perf_array, distortion_ids=(x_id, y_id, z_id), fit_key=fit_key,
                           add_bias=add_bias)
    fit_direct_correlation = evaluate_fit(fit_coefficients, distortion_array, perf_array,
                                          distortion_ids=(x_id, y_id, z_id),
                                          fit_key=fit_key, add_bias=add_bias)

    if hasattr(model_performance, 'eval_results'):
        __, __, __, perf_3d_eval, distortion_array_eval, __, __ = (
            model_performance.get_3d_distortion_perf_props(distortion_ids=(x_id, y_id, z_id),
                                                           predict_eval_flag='eval'))

        predict_eval_perf_diff_mean = np.mean(perf_3d_eval - perf_3d)
        print(f'{result_name} predict/eval perf differences: {predict_eval_perf_diff_mean}', file=log_file)

        diff_check, __ = model_performance.compare_predict_eval()
        assert np.mean(diff_check) == predict_eval_perf_diff_mean

    else:
        perf_3d_eval = perf_3d
        distortion_array_eval = distortion_array

    fit_prediction = apply_fit(fit_coefficients, distortion_array_eval, distortion_ids=(x_id, y_id, z_id),
                               fit_key=fit_key, add_bias=add_bias)

    performance_prediction_3d = build_3d_field(distortion_array[:, 0],
                                               distortion_array[:, 1],
                                               distortion_array[:, 2],
                                               fit_prediction,
                                               data_dump=False)

    eval_fit_correlation = evaluate_fit(fit_coefficients, distortion_array_eval, perf_3d_eval,
                                        distortion_ids=(x_id, y_id, z_id), fit_key=fit_key, add_bias=add_bias)
    _eval_fit_correlation = np.corrcoef(np.ravel(performance_prediction_3d), np.ravel(perf_3d_eval))[0, 1]
    assert eval_fit_correlation == _eval_fit_correlation

    p_simulate = np.clip(performance_prediction_3d, 0, 1)
    num_clipped_points = np.count_nonzero(np.where(p_simulate != performance_prediction_3d))
    total_sim_trials = len(model_performance)

    __, ideal_correlation, __, __ = get_ideal_correlation(p_simulate,
                                                          total_trials=total_sim_trials)

    mean_perf = np.mean(perf_3d)
    mean_perf_prediction = np.mean(performance_prediction_3d)
    mean_perf_eval = np.mean(perf_3d_eval)

    trials_per_experiment = len(model_performance) / len(np.ravel(perf_3d))
    perf_3d_simulated = run_binomial_accuracy_experiment(p_simulate, trials_per_experiment)

    print(f'{result_name} {x_id} {y_id} {z_id} {fit_key} performance means (performance / perf_fit_prediction'
          f'/perf_eval): {mean_perf} / {mean_perf_prediction} / {mean_perf_eval}', file=log_file)
    print(f'{fit_key} fit: \n', fit_coefficients, file=log_file)
    print(f'{fit_key} direct fit correlation: ', fit_direct_correlation,
          file=log_file)
    print(f'{fit_key} eval fit correlation: ', eval_fit_correlation,
          file=log_file)
    print(f'{fit_key} ideal fit correlation: ', ideal_correlation, '\n',
          file=log_file)

    print('dw stats (measured results)', file=log_file)
    dw_prob_sorted, dw_min_2d, dw_min_1d, dw_stats = check_durbin_watson_statistics(performance_prediction_3d,
                                                                                    perf_3d_eval,
                                                                                    x_id=x_id, y_id=y_id, z_id=z_id,
                                                                                    output_file=log_file)
    print('\n', file=log_file)

    print('dw stats (simulated results)', file=log_file)
    __, __, __, __ = check_durbin_watson_statistics(performance_prediction_3d, perf_3d_simulated,
                                                    x_id=x_id, y_id=y_id, z_id=z_id, output_file=log_file)
    print('\n', file=log_file)

    print(f'{result_name} ideal fit simulation clipped values: {num_clipped_points}, '
          f'{100 * num_clipped_points / len(np.ravel(performance_prediction_3d))}% of total', '\n', file=log_file)

    return (x_values, y_values, z_values, perf_3d, perf_3d_eval, performance_prediction_3d, perf_3d_simulated,
            eval_fit_correlation, dw_prob_sorted, dw_min_2d, dw_min_1d, dw_stats)


def analyze_perf_3d(model_performance,
                    distortion_ids=('res', 'blur', 'noise'),
                    log_file=None,
                    add_bias=True,
                    directory=None,
                    fit_key='linear',
                    standard_plots=True,
                    residual_plot=True,
                    make_residual_color_plot=True,
                    isosurf_plot=False,
                    make_simulation_plots_1d=False,
                    make_simulation_plots_2d=False,
                    log_3d_prediction=True,
                    show_plots=False,
                    show_scatter_plots=True,
                    show_1d_plots=True):

    x_id, y_id, z_id = distortion_ids
    (x_values, y_values, z_values, perf_3d, perf_3d_eval, fit_3d, perf_3d_simulated, eval_fit_correlation,
     dw_prob_sorted, dw_min_2d, dw_min_1d, dw_stats) = (get_distortion_perf_3d(model_performance,
                                                                               x_id=x_id, y_id=y_id, z_id=z_id,
                                                                               add_bias=add_bias, log_file=log_file,
                                                                               fit_key=fit_key,))

    check_histograms(perf_3d, perf_3d_eval, fit_3d, directory=directory)
    sorted_linear_scatter(fit_3d, perf_3d, directory=directory, best_fit=True, show_plots=show_scatter_plots)
    sorted_linear_scatter(fit_3d, perf_3d_eval, directory=directory, filename='predict_result_eval_scatter.png',
                          best_fit=True, show_plots=show_scatter_plots)

    sorted_linear_scatter(fit_3d, perf_3d_simulated, directory=directory,
                          filename='predict_simulated_result_scatter.png', best_fit=True,
                          xlabel='predicted accuracy (binomial simulation p-success)', ylabel='accuracy (simulated)',
                          show_plots=show_scatter_plots)

    dual_sorted_linear_scatter(fit_3d, perf_3d_eval, fit_3d, perf_3d_simulated, directory=directory,
                               filename='predict_result_eval_sim_scatter.png', xlabel_0='predicted accuracy',
                               xlabel_1='predicted accuracy (binomial simulation p-success)',
                               show_plots=show_scatter_plots)

    if standard_plots:
        compare_2d_views(perf_3d, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                         data_labels=('measured (predict)', 'fit'), az_el_combinations='all', directory=directory,
                         residual_plot=False, result_id='predict_fit_3d_proj',
                         show_plots=show_plots)

        compare_1d_views(perf_3d, fit_3d,  x_values, y_values, z_values, distortion_ids=distortion_ids,
                         data_labels=('measured (predict)', 'fit'), directory=directory,
                         result_id='mean_1d_predict',
                         show_plots=show_1d_plots)

        if not np.array_equal(perf_3d, perf_3d_eval):
            compare_2d_views(perf_3d_eval, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                             data_labels=('measured (eval)', 'fit'), az_el_combinations='all', directory=directory,
                             residual_plot=False, result_id='eval_fit_3d_proj',
                             show_plots=show_plots)

            compare_1d_views(perf_3d_eval, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                             data_labels=('measured (eval)', 'fit'), directory=directory,
                             result_id='mean_1d_eval',
                             show_plots=show_1d_plots)

        else:
            print('per_3d = perf_3d_eval for ', str(model_performance))

    if make_simulation_plots_1d:
        compare_1d_views(perf_3d_simulated, fit_3d,  x_values, y_values, z_values, distortion_ids=distortion_ids,
                         data_labels=('simulated', 'fit'), directory=directory,
                         result_id='mean_1d_simulated',
                         show_plots=show_1d_plots)

    if make_simulation_plots_2d:
        compare_2d_views(perf_3d_simulated, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                         data_labels=('simulated', 'fit'), az_el_combinations='all', directory=directory,
                         residual_plot=False, result_id='simulation_fit_3d_proj',
                         show_plots=show_plots)

    if residual_plot:
        compare_2d_views(perf_3d, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                         data_labels=('measured (predict)', 'fit'), az_el_combinations='all', directory=directory,
                         residual_plot=residual_plot, result_id='predict_fit_3d_proj')

        if not np.array_equal(perf_3d, perf_3d_eval):
            compare_2d_views(perf_3d_eval, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                             data_labels=('measured (eval)', 'fit'), az_el_combinations='all', directory=directory,
                             residual_plot=residual_plot, result_id='eval_fit_3d_proj',
                             show_plots=show_plots)

    if make_residual_color_plot:
        residual_color_plot(perf_3d, fit_3d, x_values, y_values, z_values, distortion_ids=distortion_ids,
                            directory=directory)

    if isosurf_plot:
        save_name = f'{str(model_performance)}_fit_isosurf.png'
        iso_save_dir = Path(directory, 'isosurf')
        if not iso_save_dir.is_dir():
            Path.mkdir(iso_save_dir)
        plot_isosurf(fit_3d, x_values, y_values, z_values,
                     levels=[-0.3, 0, 0.3], save_name=save_name, save_dir=iso_save_dir,
                     az_el_combinations='all',
                     show_plots=show_plots)

    if log_3d_prediction:
        perf_prediction_dir = Path(directory, REL_PATHS['perf_prediction'])
        log_3d_perf_prediction(fit_3d, x_values, y_values, z_values, perf_prediction_dir, distortion_ids=distortion_ids)

    log_fit_stats(fit_stats=dw_stats, directory=directory, eval_fit_correlation=eval_fit_correlation, fit_key=fit_key)

    return round(eval_fit_correlation, 3), round(dw_prob_sorted, 3), round(dw_min_2d, 3), round(dw_min_1d, 3)


def check_histograms(distortion_performance_predict, distortion_performance_eval, performance_fit, directory=None,
                     show_plots=False):

    plt.figure()
    plt.hist(np.ravel(distortion_performance_predict))
    plt.ylabel('occurrences')
    plt.xlabel('accuracy')
    if directory:
        plt.savefig(Path(directory, 'hist_performance_predict.png'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.ravel(distortion_performance_eval))
    plt.ylabel('occurrences')
    plt.xlabel('accuracy')
    if directory:
        plt.savefig(Path(directory, 'hist_performance.png'))
    if show_plots:
        plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.ravel(performance_fit))
    plt.ylabel('occurrences')
    plt.xlabel('accuracy')
    if directory:
        plt.savefig(Path(directory, 'hist_fit.png'))
    if show_plots:
        plt.show()
    plt.close()

    residuals = performance_fit - distortion_performance_eval

    plt.figure()
    plt.hist(np.ravel(residuals))
    plt.ylabel('occurrences')
    plt.xlabel('residual')
    if directory:
        plt.savefig(Path(directory, 'hist_residuals.png'))
    if show_plots:
        plt.show()
    plt.close()


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

    with wandb.init(project=WANDB_PID, job_type='analyze_test_result') as run:
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

    with wandb.init(project=WANDB_PID, job_type='analyze_test_result') as run:

        for (artifact_id, identifier) in result_id_pairs:

            artifact_id, __ = construct_artifact_id(artifact_id)

            performance_result, __ = _fetch_model_distortion_performance_result(run, artifact_id, identifier,
                                                                                distortion_ids, make_dir=make_dir)
            if output_type == 'list':
                performance_results.append(performance_result)
            elif output_type == 'dict':
                performance_results[artifact_id] = performance_result

    return performance_results


def performance_fit_summary_text_dump(fit_summary, header='eval fit correlation / durbin-watson summary', file=None,
                                      print_to_console=False):

    print(header, file=file)
    if print_to_console:
        print(header)

    for key, val in fit_summary.items():
        print(f'{key}: {val}', file=file)

        if print_to_console:
            print(f'{key}: {val}')


def log_3d_perf_prediction(perf_prediction_3d, x_values, y_values, z_values, directory, distortion_ids=None):

    if not Path(directory).is_dir():
        Path.mkdir(directory)

    np.savez(Path(directory, STANDARD_PERFORMANCE_PREDICTION_FILENAME),
             perf_prediction_3d=perf_prediction_3d,
             x=x_values,
             y=y_values,
             z=z_values,
             distortion_ids=distortion_ids)


def log_fit_stats(fit_stats, directory, eval_fit_correlation=None, fit_key=None):

    if fit_key:
        fit_stats['fit_key'] = fit_key

    if eval_fit_correlation is not None:
        fit_stats['eval_fit_correlation'] = float(eval_fit_correlation)

    with open(Path(directory, STANDARD_FIT_STATS_FILENAME), 'w') as file:
        dump(fit_stats, file)


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
