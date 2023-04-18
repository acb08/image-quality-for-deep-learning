from src.analysis.distortion_performance import get_model_distortion_performance_result
from src.analysis.plot import AXIS_LABELS
from src.utils.definitions import ROOT_DIR, REL_PATHS
from src.utils.functions import get_config
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from src.analysis.plot import analyze_plot_perf_1d_multi_result


def compare_scans(test_result_identifiers, distortion_id, manual_name):

    distortion_ids = (distortion_id, )
    results = []

    save_dir = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['distortion_scan'], manual_name)
    if not save_dir.is_dir():
        Path.mkdir(save_dir)

    for i, (result_id, identifier) in enumerate(test_result_identifiers):
        result, __ = get_model_distortion_performance_result(result_id=result_id,
                                                             identifier=identifier,
                                                             distortion_ids=distortion_ids,
                                                             make_dir=False)
        results.append(result)

    analyze_plot_perf_1d_multi_result(results,
                                      distortion_ids=distortion_ids,
                                      directory=save_dir,
                                      identifier=None,
                                      legend_loc='best',
                                      log_file=None,
                                      plot_together=False,
                                      show_plots=True,
                                      single_legend=True,
                                      perform_fits=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='compare_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_scan_compare'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    compare_scans(**run_config)



