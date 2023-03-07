import numpy as np
import argparse
from pathlib import Path
from src.utils.functions import get_config, log_config
from src.analysis.compare_correlate_results import get_compare_dir
from src.obj_det_analysis.distortion_performance_od import get_obj_det_distortion_perf_result, \
    flatten_axis_combinations_from_cfg, flatten_axes_from_cfg
from src.utils.definitions import WANDB_PID
from src.utils.functions import construct_artifact_id
import wandb
from src.analysis.plot import plot_1d_from_3d


def get_multiple_od_distortion_performance_results(result_id_pairs,
                                                   output_type='list'):

    if output_type == 'list':
        performance_results = []
    elif output_type == 'dict':
        performance_results = {}
    else:
        raise Exception('invalid output_type')

    with wandb.init(project=WANDB_PID, job_type='analyze_test_result') as run:

        for artifact_id, identifier in result_id_pairs:

            artifact_id, __ = construct_artifact_id(artifact_id)
            distortion_performance_result, __ = get_obj_det_distortion_perf_result(result_id=artifact_id,
                                                                                   identifier=identifier,
                                                                                   make_dir=False,
                                                                                   run=run)

            if output_type == 'list':
                performance_results.append(distortion_performance_result)
            elif output_type == 'dict':
                performance_results[artifact_id] = distortion_performance_result\

    return performance_results


def main(config):

    wandb.login()

    test_result_identifiers = config['test_result_identifiers']
    distortion_performance_results = get_multiple_od_distortion_performance_results(
        result_id_pairs=test_result_identifiers)
    flatten_axes = flatten_axes_from_cfg(config)
    flatten_axis_combinations = flatten_axis_combinations_from_cfg(config)

    output_dir = get_compare_dir(test_result_identifiers, manual_name=config['manual_name'])
    log_config(output_dir=output_dir, config=config)

    performance_dict_3d = {}

    res_vals = None
    blur_vals = None
    noise_vals = None

    for distortion_performance_result in distortion_performance_results:

        _res_vals, _blur_vals, _noise_vals, _map3d, _parameter_array, _perf_array, _full_extract = (
            distortion_performance_result.get_3d_distortion_perf_props(distortion_ids=('res', 'blur', 'noise')))
        key = str(distortion_performance_result)
        performance_dict_3d[key] = _map3d

        if res_vals is None:
            res_vals = _res_vals
        else:
            assert np.array_equal(_res_vals, res_vals)
        if blur_vals is None:
            blur_vals = _blur_vals
        else:
            assert np.array_equal(blur_vals, _blur_vals)
        if noise_vals is None:
            noise_vals = _noise_vals
        else:
            assert np.array_equal(noise_vals, _noise_vals)

    plot_1d_from_3d(perf_dict_3d=performance_dict_3d,
                    x_vals=res_vals,
                    y_vals=blur_vals,
                    z_vals=noise_vals,
                    distortion_ids=('res', 'blur', 'noise'),
                    flatten_axis_combinations=flatten_axis_combinations,
                    show_plots=True,
                    plot_together=False,
                    directory=output_dir,
                    ylabel='mAP',
                    legend=False,
                    y_lim_bottom=-0.03,
                    y_lim_top=0.65)

    if flatten_axes is not None:
        pass

    return distortion_performance_results


if __name__ == '__main__':

    config_name = 'fr_pt_n_scan.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_name, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'compare_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    main(run_config)
