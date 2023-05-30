import numpy as np
import argparse
from pathlib import Path
from src.obj_det_analysis.load_multiple_od_results import get_multiple_od_distortion_performance_results
from src.utils.functions import get_config, log_config
from src.analysis.compare_correlate_results import get_compare_dir
from src.obj_det_analysis.distortion_performance_od import flatten_axis_combinations_from_cfg, flatten_axes_from_cfg, fit_keys_from_cfg
import wandb
from src.analysis.plot import plot_1d_from_3d, compare_2d_mean_views
from src.analysis.analysis_functions import get_sub_dir_and_log_filename, build_3d_field
from src.analysis.fit import fit, evaluate_fit, apply_fit
from src.analysis import plot
from src.analysis.aic import akaike_info_criterion
from src.obj_det_analysis.distortion_performance_composite_od import get_composite_performance_result_od
from src.utils.classes import PseudoArgs


def log_averages(performance_dict, output_dir):

    with open(Path(output_dir, 'mean_performances.txt'), 'w') as log_file:

        for key, map_3d in performance_dict.items():
            mean_perf = float(np.mean(map_3d))
            print(f'{key}, mAP average over distortion space: {round(mean_perf, 3)}', file=log_file)
        print('\n')


def main(config):

    wandb.login()

    test_result_identifiers = config['test_result_identifiers']

    all_results = []

    distortion_performance_results = get_multiple_od_distortion_performance_results(
        result_id_pairs=test_result_identifiers)
    flatten_axes = flatten_axes_from_cfg(config)
    flatten_axis_combinations = flatten_axis_combinations_from_cfg(config)

    if 'plot_together' in config.keys():
        plot_together = config['plot_together']
    else:
        plot_together = False

    if 'show_plots' in config.keys():
        show_plots = config['show_plots']
    else:
        show_plots = False

    if 'log_average_performances' in config.keys():
        log_average_performances = config['log_average_performances']
    else:
        log_average_performances = True

    y_lim_bottom, y_lim_top = None, None
    if 'y_limits' in config.keys():
        y_limits = config['y_limits']
        if y_limits is not None:
            y_lim_bottom, y_lim_top = config['y_limits']

    if 'fit_keys' in config.keys():  # default is to not generate fits
        fit_keys = fit_keys_from_cfg(config)
    else:
        fit_keys = None

    if 'allow_differing_distortions' in config.keys():
        allow_differing_distortions = config['allow_differing_distortions']
    else:
        allow_differing_distortions = False

    if 'composite_config_filenames' in config.keys():
        composite_config_filenames = config['composite_config_filenames']

        composite_results = []

        for composite_config_filename in composite_config_filenames:
            pseudo_args = PseudoArgs(config_dir=config['composite_config_dir'],
                                     config_name=composite_config_filename)
            composite_config = get_config(pseudo_args)
            composite_performance_result_od, __ = get_composite_performance_result_od(composite_config)
            composite_results.append(composite_performance_result_od)

        all_results.extend(composite_results)

    all_results.extend(distortion_performance_results)

    output_dir = get_compare_dir(test_result_identifiers, manual_name=config['manual_name'])
    log_config(output_dir=output_dir, config=config)

    performance_dict_3d = {}

    res_vals = None
    blur_vals = None
    noise_vals = None

    if fit_keys is not None:
        fitting_arrays = {}
    else:
        fitting_arrays = None

    if 'single_legend' in config.keys():
        single_legend = config['single_legend']
    else:
        single_legend = True

    for result in all_results:

        _res_vals, _blur_vals, _noise_vals, _map3d, _parameter_array, _perf_array, _full_extract = (
            result.get_3d_distortion_perf_props(distortion_ids=('res', 'blur', 'noise')))
        key = str(result)
        performance_dict_3d[key] = _map3d

        if fitting_arrays is not None:
            fitting_arrays[key] = (_parameter_array, _perf_array)

        if res_vals is None:
            res_vals = _res_vals

        if blur_vals is None:
            blur_vals = _blur_vals

        if noise_vals is None:
            noise_vals = _noise_vals

        if not allow_differing_distortions:
            assert np.array_equal(noise_vals, _noise_vals)
            assert np.array_equal(blur_vals, _blur_vals)
            assert np.array_equal(_res_vals, res_vals)

    if flatten_axis_combinations is not None:
        plot_1d_from_3d(perf_dict_3d=performance_dict_3d,
                        x_vals=res_vals,
                        y_vals=blur_vals,
                        z_vals=noise_vals,
                        distortion_ids=('res', 'blur', 'noise'),
                        flatten_axis_combinations=flatten_axis_combinations,
                        show_plots=show_plots,
                        plot_together=plot_together,
                        directory=output_dir,
                        ylabel='mAP',
                        legend=True,
                        y_lim_bottom=y_lim_bottom,
                        y_lim_top=y_lim_top,
                        single_legend=single_legend
                        )

    if flatten_axes is not None:

        compare_2d_mean_views(f0=performance_dict_3d, f1=None,
                              x_vals=res_vals, y_vals=blur_vals, z_vals=noise_vals,
                              distortion_ids=('res', 'blur', 'noise'),  # flatten_axes=_flatten_axes,
                              directory=output_dir,
                              perf_metric='mAP',
                              show_plots=show_plots,
                              az_el_combinations='all',
                              flatten_axes=flatten_axes)

    if fit_keys is not None:

        assert len(distortion_performance_results) == 2
        predict_key = str(distortion_performance_results[0])
        eval_key = str(distortion_performance_results[1])

        predict_parameter_array, predict_performance_array = fitting_arrays[predict_key]
        eval_parameter_array, eval_performance_array = fitting_arrays[eval_key]
        map3d_measured = performance_dict_3d[eval_key]

        sub_dir, log_filename = get_sub_dir_and_log_filename(output_dir, '3d')

        with open(Path(sub_dir, log_filename), 'w') as output_file:

            print('predict result: ', predict_key, file=output_file)
            print('eval result: ', eval_key, '\n', file=output_file)

            for fit_key in fit_keys:

                fit_sub_dir, __ = get_sub_dir_and_log_filename(sub_dir, fit_key)
                fit_coefficients = fit(predict_parameter_array, predict_performance_array,
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
                                                    data_dump=False
                                                    )
                predict_fit_correlation = evaluate_fit(fit_coefficients,
                                                       predict_parameter_array,
                                                       predict_performance_array,
                                                       distortion_ids=('res', 'blur', 'noise'),
                                                       fit_key=fit_key,
                                                       add_bias=False,  # only applies to linear fits
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
                                           show_plots=False)

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

    if log_average_performances:
        log_averages(performance_dict=performance_dict_3d, output_dir=output_dir)

    return distortion_performance_results


if __name__ == '__main__':

    config_name = 'v8l-fr-ext-m2_composite-res-4-1.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_name, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'compare_configs'),
                        help="configuration file directory")
    parser.add_argument('--composite_config_dir',
                        default=Path(Path(__file__).parents[0], 'composite_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    composite_config_dir = str(args_passed.composite_config_dir)

    run_config = get_config(args_passed)
    run_config['composite_config_dir'] = composite_config_dir

    main(run_config)
