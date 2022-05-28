import copy
import json
from src.d04_analysis.analysis_functions import build_3d_field
from src.d04_analysis.distortion_performance import get_multiple_model_distortion_performance_results
from src.d04_analysis.plot import analyze_plot_results_together
from src.d00_utils.functions import get_config, log_config, increment_suffix
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import itertools


def build_screen_performance_arrays(performance_results, x_id='res', y_id='blur', z_id='noise',
                                    distortion_array_check=None):
    result_ids = []
    model_dataset_id_pairs = []
    manual_ids = []
    result_list = []

    # start by organizing performance_results by (model_id, dataset_id) tuples and checking for results that are
    # duplicates (i.e. instances of a model/dataset combo being tested twice)
    for performance_result in performance_results:

        model_id = performance_result.model_id
        dataset_id = performance_result.dataset_id
        result_id = performance_result.result_id
        manual_id = str(performance_result)

        id_pair = (model_id, dataset_id)

        if id_pair in model_dataset_id_pairs:
            duplicate_index = model_dataset_id_pairs.index(id_pair)
            duplicate_id = result_ids[duplicate_index]
            duplicate_manual_id = manual_ids[duplicate_index]
            print(f'result id {result_id} / {manual_id} appears to be a duplicate of {duplicate_id} / '
                  f'{duplicate_manual_id}')

        model_dataset_id_pairs.append(id_pair)
        result_ids.append(result_id)
        manual_ids.append(manual_id)

        try:
            x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = \
                performance_result.get_3d_distortion_perf_props(distortion_ids=(x_id, y_id, z_id))
        except ValueError:
            x = performance_result.distortions[x_id]
            y = performance_result.distortions[y_id]
            z = performance_result.distortions[z_id]
            accuracy_vector = performance_result.perf_predict_top_1_array

            x_values, y_values, z_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(x, y, z,
                                                                                                     accuracy_vector,
                                                                                                     data_dump=True)
        # verify that all results come from test datasets with identical distortion values
        if distortion_array_check is None:
            distortion_array_check = distortion_array

        elif not np.array_equal(distortion_array_check, distortion_array):
            raise Exception('performance results contain results with differing distortion parameters')

        result_list.append(np.ravel(perf_array))

    return result_list, model_dataset_id_pairs, result_ids, manual_ids, distortion_array_check


def get_performance_correlations(performance_results, performance_results_second_set=None,
                                 x_id='res', y_id='blur', z_id='noise'):

    if not performance_results_second_set:
        results_0, model_dataset_id_pairs_0, result_ids_0, manual_ids_0, __ = build_screen_performance_arrays(
            performance_results, x_id=x_id, y_id=y_id, z_id=z_id)
        results_1 = copy.deepcopy(results_0)
        model_dataset_id_pairs_1 = copy.deepcopy(model_dataset_id_pairs_0)
        result_ids_1 = copy.deepcopy(result_ids_0)
        manual_ids_1 = copy.deepcopy(manual_ids_0)

    else:
        results_0, model_dataset_id_pairs_0, result_ids_0, manual_ids_0, distortion_array_check = (
            build_screen_performance_arrays(performance_results, x_id=x_id, y_id=y_id, z_id=z_id))

        results_1, model_dataset_id_pairs_1, result_ids_1, manual_ids_1, __ = build_screen_performance_arrays(
            performance_results_second_set, x_id=x_id, y_id=y_id, z_id=z_id,
            distortion_array_check=distortion_array_check)

    correlations = np.zeros((len(results_0), len(results_1)))
    identifiers = {}
    array_index_id_dict = {}

    for i, r0 in enumerate(results_0):
        for j, r1 in enumerate(results_1):

            correlation = np.corrcoef(r0, r1)[0, 1]
            correlations[i, j] = correlation

            model_dataset_id_pair_0 = model_dataset_id_pairs_0[i]
            model_dataset_id_pair_1 = model_dataset_id_pairs_1[j]
            result_id_0 = result_ids_0[i]
            result_id_1 = result_ids_1[j]
            manual_id_0 = manual_ids_0[i]
            manual_id_1 = manual_ids_1[j]

            array_index_id_dict[f'{i}-{j}'] = {
                'paired_model_dataset_id_pairs': [model_dataset_id_pair_0, model_dataset_id_pair_1],
                'result_id_pairs': [result_id_0, result_id_1],
                'manual_id_pairs': [manual_id_0, manual_id_1]
            }

    identifiers['label_lists'] = {
        'model_dataset_id_pairs': [model_dataset_id_pairs_0, model_dataset_id_pairs_1],
        'result_ids': [result_ids_0, result_ids_1],
        'manual_ids': [manual_ids_0, manual_ids_1]
    }

    return correlations, identifiers


def get_correlation_result_dir(result_id_pairs, parent_dir='default', overwrite=True, suffix='v2', manual_name=None):

    if parent_dir == 'default':
        parent_dir = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['perf_correlations'])

    if not parent_dir.is_dir():
        Path.mkdir(parent_dir)

    if manual_name:
        new_dir_name = manual_name
    else:
        new_dir_name = None
        for pair in result_id_pairs:
            result_name = pair[0]
            result_num_string = result_name[:4]
            if not new_dir_name:
                new_dir_name = result_num_string
            else:
                new_dir_name = f'{new_dir_name}-{result_num_string}'

    new_dir = Path(parent_dir, new_dir_name)

    if not new_dir.is_dir():
        Path.mkdir(new_dir)
        return new_dir
    elif overwrite:
        return new_dir
    else:
        new_dir = Path(new_dir, suffix)

    if not new_dir.is_dir():
        Path.mkdir(new_dir)
        return new_dir

    else:  # recursion #nbd #nextlinustorvalds
        new_suffix = increment_suffix(suffix)
        return get_correlation_result_dir(result_id_pairs, overwrite=overwrite, suffix=new_suffix)


def analyze_correlations(config):

    test_result_identifiers = config['test_result_identifiers']
    different_result_identifiers = config['different_test_result_identifiers']
    overwrite = config['overwrite']
    manual_name = config['manual_name']
    analyze_3d = config['analyze_3d']
    analyze_pairwise = config['analyze_pairwise']

    dir_name_result_identifiers = test_result_identifiers
    if different_result_identifiers:
        dir_name_result_identifiers.extend(different_result_identifiers)

    output_dir = get_correlation_result_dir(dir_name_result_identifiers,
                                            overwrite=overwrite, manual_name=manual_name)

    performance_results = get_multiple_model_distortion_performance_results(test_result_identifiers)
    different_performance_results = None
    if different_result_identifiers:
        different_performance_results = get_multiple_model_distortion_performance_results(different_result_identifiers)

    if analyze_3d:
        correlations, identifiers = get_performance_correlations(performance_results, different_performance_results)
        log_correlation_matrix_txt(correlations, identifiers, output_dir=output_dir)
        label_dict = identifiers['label_lists']
        make_log_dataframes(correlations, label_dict, output_dir=output_dir)
        log_config(output_dir, config)

    if analyze_pairwise:
        all_performance_results = performance_results
        if different_result_identifiers:
            all_performance_results.extend(different_performance_results)
        analyze_pairwise_1d_2d(all_performance_results)


def make_log_dataframes(correlations, label_dict, output_dir=None):

    dfs = {}
    for label_key, label_list_pair in label_dict.items():
        index_labels = label_list_pair[0]
        col_labels = label_list_pair[1]
        df = pd.DataFrame(correlations, index=index_labels, columns=col_labels)
        dfs[label_key] = df
        if output_dir:
            with open(Path(output_dir, f'{label_key}.tex'), 'w') as file:
                print(df.to_latex(index=True), file=file)


def log_correlation_matrix_txt(correlations, identifiers, output_dir=None):

    with open(Path(output_dir, 'correlations.txt'), 'w') as file:
        print('cross correlation matrix: \n', correlations, '\n', file=file)
        print(json.dumps(identifiers, indent=1), '\n', file=file)


def analyze_pairwise_1d_2d(model_results, directory='default', make_subdirectories=True,
                           create_log_files=True, legend_loc='best'):

    if directory == 'default':
        directory = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['pairwise'])

    for model_result_pair in itertools.combinations(model_results, 2):

        analyze_plot_results_together(model_result_pair, directory=directory, make_subdir=make_subdirectories,
                                      dim_tag='2d', legend_loc=legend_loc, create_log_file=create_log_files,
                                      pairwise_analysis=True)
        analyze_plot_results_together(model_result_pair, directory=directory, make_subdir=make_subdirectories,
                                      dim_tag='1d', legend_loc=legend_loc, create_log_file=create_log_files,
                                      pairwise_analysis=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_name', default='cross_correlation_config.yml', help='config filename to be used')
    parser.add_argument('--config_name', default='pl_fr_models_megasets_1_2.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'cross_correlation_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    analyze_correlations(run_config)
