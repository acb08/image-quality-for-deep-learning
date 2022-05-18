from src.d04_analysis.analysis_functions import get_performance_correlations
from src.d04_analysis.distortion_performance import get_model_distortion_performance_result
from src.d00_utils.functions import get_config
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def concatenate_performance_results(result_id_pairs):
    performance_results = []
    for (artifact_id, identifier) in result_id_pairs:
        performance_result, __ = get_model_distortion_performance_result(artifact_id, identifier)
        performance_results.append(performance_result)
    return performance_results


def main(config):

    result_id_pairs = config['result_id_pairs']
    performance_results = concatenate_performance_results(result_id_pairs)
    correlations, alt_id_correlations = get_performance_correlations(performance_results)

    _print_correlation_dict(correlations)
    _print_correlation_dict(alt_id_correlations)

    return correlations, alt_id_correlations


def _print_correlation_dict(correlations):
    for key, item in correlations.items():
        print(key, item)


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

    return cross_correlation_matrix, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='cross_correlation_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'cross_correlation_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _correlations, _alt_id_correlations = main(run_config)

    _cross_correlation_matrix, _df = make_result_table(_alt_id_correlations)
