# from src.obj_det_analysis.distortion_performance_composite_od import get_composite_performance_result_od
import argparse
from src.utils.functions import get_config
from pathlib import Path
from src.obj_det_analysis.distortion_performance_composite_od import get_composite_performance_result_od
from src.obj_det_analysis.load_multiple_od_results import get_multiple_od_distortion_performance_results
from src.analysis.fit import fit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'composite_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    composite_config = get_config(args_passed)

    config_name = 'v8l-fr-ext_fr90_m1-v-m2.yml'
    multi_result_parser = argparse.ArgumentParser()
    multi_result_parser.add_argument('--config_name', default=config_name, help='config filename to be used')
    multi_result_parser.add_argument('--config_dir',
                                     default=Path(Path(__file__).parents[0], 'compare_configs'),
                                     help="configuration file directory")
    multi_result_args_passed = multi_result_parser.parse_args()
    multi_result_config = get_config(multi_result_args_passed)

    test_result_identifiers = multi_result_config['test_result_identifiers']
    distortion_performance_results = get_multiple_od_distortion_performance_results(
        result_id_pairs=test_result_identifiers)

    composite_result, output_dir = get_composite_performance_result_od(composite_config)

    composite_performance_od = composite_result
    predict_data = composite_performance_od.get_3d_distortion_perf_props(predict_eval_flag='predict')
    res_vals, blur_vals, noise_vals, map_3d_predict, predict_param_array, predict_perf_array, __ = predict_data

    distortion_performance_result = distortion_performance_results[0]

    _res_vals, _blur_vals, _noise_vals, _map3d, _parameter_array, _perf_array, _full_extract = (
        distortion_performance_result.get_3d_distortion_perf_props(distortion_ids=('res', 'blur', 'noise')))

    fit_key = 'giqe3_b2n2'

    fit_coefficients = fit(_parameter_array, _perf_array,
                           distortion_ids=('res', 'blur', 'noise'),
                           fit_key=fit_key,
                           add_bias=False  # only applies to linear fits
                           )
    fit_coefficients_composite = fit(predict_param_array, predict_perf_array,
                                     distortion_ids=('res', 'blur', 'noise'),
                                     fit_key=fit_key,
                                     add_bias=False  # only applies to linear fits
                                     )
