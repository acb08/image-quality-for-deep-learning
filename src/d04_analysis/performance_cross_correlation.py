from src.d04_analysis.analysis_functions import get_performance_correlations
from src.d04_analysis.distortion_performance import get_model_distortion_performance_result
from src.d00_utils.functions import get_config, log_config
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _increment_suffix(suffix):
    suffix_num = int(suffix[1:])
    suffix_num += 1
    new_suffix = f'v{suffix_num}'
    return new_suffix


def get_correlation_result_dir(result_id_pairs, overwrite=True, suffix='v2'):

    parent_dir = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['perf_correlations'])
    if not parent_dir.is_dir():
        Path.mkdir(parent_dir)

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

    else:
        new_suffix = _increment_suffix(suffix)
        return get_correlation_result_dir(result_id_pairs, overwrite=overwrite, suffix=new_suffix)


def concatenate_performance_results(result_id_pairs):
    performance_results = []
    for (artifact_id, identifier) in result_id_pairs:
        performance_result, __ = get_model_distortion_performance_result(artifact_id, identifier)
        performance_results.append(performance_result)
    return performance_results


def make_result_table(tuple_keyed_dict):

    # keys = list(tuple_keyed_dict.keys())
    left_ids = []
    right_ids = []
    for (left_id, right_id) in tuple_keyed_dict.keys():
        left_ids.append(left_id)
        right_ids.append(right_id)

    # check that left_ids and right_ids contain the same entries
    membership_equality = set(left_ids) == set(right_ids)
    if membership_equality:
        id_set = set(left_ids)
    else:
        raise Exception('left/right membership of tuple keys not symmetric, i.e "PC LOAD LETTER"')

    cross_correlation_matrix = np.zeros((len(id_set), len(id_set)))
    index_labels = []
    for i, col_id in enumerate(list(id_set)):
        index_labels.append(col_id)
        for j, row_id in enumerate(list(id_set)):
            try:
                cross_correlation_matrix[i, j] = tuple_keyed_dict[(row_id, col_id)]
            except KeyError:
                cross_correlation_matrix[i, j] = tuple_keyed_dict[(col_id, row_id)]

    col_labels = [f'{label[:5]}...' for label in index_labels]
    df = pd.DataFrame(cross_correlation_matrix, index=index_labels, columns=col_labels)

    return cross_correlation_matrix, df, index_labels


def analyze_correlations(config):

    result_id_pairs = config['result_id_pairs']
    overwrite = config['overwrite']

    output_dir = get_correlation_result_dir(result_id_pairs, overwrite=overwrite)

    performance_results = concatenate_performance_results(result_id_pairs)
    correlations, alt_id_correlations = get_performance_correlations(performance_results)
    # correlation matrix row/col order dependent on the labels used (because of how set(some_list) elements are ordered)
    cross_correlation_matrix, df_0, idx_labels_0 = make_result_table(correlations)
    cross_correlation_matrix_check, df_1, idx_labels_1 = make_result_table(alt_id_correlations)

    equal_arrays = np.array_equal(cross_correlation_matrix, cross_correlation_matrix_check)
    if not equal_arrays:
        print(cross_correlation_matrix - cross_correlation_matrix_check)
        print(np.sum(cross_correlation_matrix_check - cross_correlation_matrix))

    with open(Path(output_dir, 'correlations.txt'), 'w') as file:
        print('cross correlation matrix: \n', cross_correlation_matrix, '\n', file=file)
        print('long labels: ', idx_labels_0, '\n', file=file)

    with open(Path(output_dir, 'correlations_alt_id.txt'), 'w') as file:
        print('cross correlation matrix check: \n', cross_correlation_matrix_check, '\n', file=file)
        print('short labels: ', idx_labels_1, '\n', file=file)

    with open(Path(output_dir, 'correlation_table.tex'), 'w') as file:
        print(df_0.to_latex(index=True), file=file)

    with open(Path(output_dir, 'correlation_table_short_labels.tex'), 'w') as file:
        print(df_1.to_latex(index=True), file=file)

    log_config(output_dir, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='cross_correlation_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'cross_correlation_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    analyze_correlations(run_config)
