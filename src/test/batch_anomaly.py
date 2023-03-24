import argparse
from src.utils.functions import get_config
from src.test.test_obj_det import test_detection_model
from src.obj_det_analysis.analysis_tools import calculate_aggregate_results
from pathlib import Path
import numpy as np


def output_stats(outputs):

    num_predicts = 0
    total_confidence = 0

    for key, output in outputs.items():
        num_predicts += len(output['labels'])
        score = np.sum(output['scores'])
        total_confidence += score

    avg_confidence = total_confidence / num_predicts

    return num_predicts, avg_confidence


def result_compare(results, make_plots=False, diagnostic_details=False):

    map_vals = {}
    stats = {}

    for key, result in results.items():

        __, avg_precision_vals, __, __, __ = calculate_aggregate_results(result['outputs'],
                                                                         result['targets'],
                                                                         make_plots=make_plots,
                                                                         return_diagnostic_details=diagnostic_details)
        map_val = np.mean(avg_precision_vals)
        map_vals[key] = map_val

        stats[key] = output_stats(result['outputs'])

    return map_vals, stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='test_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'test_configs_detection'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    batch_sizes = [1, 2, 4, 6, 8]

    test_results = test_detection_model(config=run_config,
                                        batch_sweep=True,
                                        batch_sizes=batch_sizes)

    _map_vals, _stats = result_compare(test_results)

    print(_map_vals)
    print(_stats)

