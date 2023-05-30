import numpy as np
import argparse
from pathlib import Path
from src.obj_det_analysis.classes import ModelDistortionPerformanceResultOD, _PreProcessedDistortionPerformanceProps
from src.utils.functions import get_config, load_dataset_and_result, get_processed_props_artifact_id, \
    construct_artifact_id
from src.utils import definitions
import wandb
from src.analysis import plot
from src.analysis.analysis_functions import get_sub_dir_and_log_filename, build_3d_field
from src.analysis.fit import fit, evaluate_fit, apply_fit


def get_obj_det_distortion_perf_result(result_id=None, identifier=None, config=None,
                                       distortion_ids=('res', 'blur', 'noise'), make_dir=True, run=None,
                                       report_time=False,
                                       pre_processed_artifact=False):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    if config is not None:
        if 'pre_processed_artifact' in config.keys():
            pre_processed_artifact = config['pre_processed_artifact']

    output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['analysis'], result_id)
    if make_dir and not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    if pre_processed_artifact:
        print('loading pre-processed wandb artifact')
        outputs = {}  # hack to running out of memory on faster-rcnn result
        targets = {}
        distortion_performance_result = load_processed_props(result_id=result_id,
                                                             identifier=identifier,
                                                             outputs=outputs,
                                                             targets=targets)
        return distortion_performance_result, output_dir

    if run is None:
        with wandb.init(project=definitions.WANDB_PID, job_type='analyze_test_result') as run:
            print('loading dataset and result (new wandb run)')
            dataset, result, dataset_id = load_dataset_and_result(run=run, result_id=result_id)
    else:
        print('loading dataset and result (existing wandb run)')
        dataset, result, dataset_id = load_dataset_and_result(run=run, result_id=result_id)


    # else:

    distortion_performance_result = ModelDistortionPerformanceResultOD(
        dataset=dataset,
        result=result,
        convert_to_std=True,
        result_id=result_id,
        identifier=identifier,
        report_time=report_time
        )

    return distortion_performance_result, output_dir


def load_processed_props(result_id, outputs, targets, identifier=None, predict_eval_flag='predict'):

    with wandb.init(project=definitions.WANDB_PID, job_type='load_performance_properties') as run:

        artifact_id = get_processed_props_artifact_id(result_id)
        artifact_id, stem = construct_artifact_id(artifact_id)
        artifact = run.use_artifact(artifact_id)
        artifact_dir = artifact.download()
        full_artifact_path = Path(artifact_dir, definitions.STANDARD_PROCESSED_DISTORTION_PERFORMANCE_PROPS_FILENAME)

        distortion_performance_result = _PreProcessedDistortionPerformanceProps(
            processed_props_path=full_artifact_path,
            result_id=result_id,
            outputs=outputs,
            targets=targets,
            identifier=identifier,
            predict_eval_flag=predict_eval_flag,
            ignore_vc_hashes=True,

        )

    return distortion_performance_result


def flatten_axes_from_cfg(config):

    default = 0, 1, 2

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
            flatten_axis_combinations = default

        if flatten_axis_combinations is not None:
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


if __name__ == '__main__':

    _REPORT_TIME = True

    ide_config_name = '000-m1.yml'

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

    _distortion_performance_result, _output_dir = get_obj_det_distortion_perf_result(config=run_config,
                                                                                     report_time=_REPORT_TIME)

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
                                   show_plots=True,
                                   flatten_axes=_flatten_axes)

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
                plot.compare_2d_mean_views(f0=_map3d, f1=_direct_fit_prediction_3d,
                                           data_labels=('measured', 'fit'),
                                           x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                                           distortion_ids=('res', 'blur', 'noise'),  # flatten_axes=_flatten_axes,
                                           directory=_fit_sub_dir,
                                           perf_metric='mAP',
                                           az_el_combinations='all',
                                           show_plots=False)

                plot.compare_2d_slice_views(f0=_map3d, f1=_direct_fit_prediction_3d,
                                            data_labels=('measured', 'fit'),
                                            x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                                            distortion_ids=('res', 'blur', 'noise'),  # flatten_axes=_flatten_axes,
                                            directory=_fit_sub_dir,
                                            perf_metric='mAP',
                                            az_el_combinations='mini',
                                            show_plots=False,
                                            sub_dir_per_az_el=True,
                                            )

                plot.compare_1d_views(f0=_map3d, f1=_direct_fit_prediction_3d,
                                      data_labels=('measured', 'fit'),
                                      x_vals=_res_vals, y_vals=_blur_vals, z_vals=_noise_vals,
                                      flatten_axis_combinations=_flatten_axis_combinations,
                                      result_id='3d_1d_projection',
                                      directory=_fit_sub_dir,
                                      show_plots=True,
                                      plot_together=True,
                                      ylabel='mAP')

                print(f'{_fit_key} fit: \n', _fit_coefficients, file=_output_file)
                print(f'{_fit_key} direct fit correlation: ', _fit_correlation, '\n',
                      file=_output_file)


