from src.d04_analysis.distortion_performance import get_model_distortion_performance_result
from src.d04_analysis.plot import AXIS_LABELS
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def chance_perf_distortion(result, distortion_id, save=False, mpc=True):

    all_labels = np.unique(result.labels)
    chance_perf = 1 / len(all_labels)
    distortion_values, accuracies = result.conditional_accuracy(distortion_id, per_class=False)
    if mpc:
        ___, mpc_accuracies = result.conditional_accuracy(distortion_id, per_class=True)

    plt.figure()
    plt.plot(distortion_values, accuracies, label='pre-trained model accuracy')
    if mpc:
        plt.plot(distortion_values, mpc_accuracies, label='pre-trained model mpc accuracy')
    plt.plot(distortion_values, chance_perf * np.ones_like(distortion_values),
             label='chance performance level', linestyle='dotted')
    plt.xlabel(AXIS_LABELS[distortion_id])
    plt.ylabel(AXIS_LABELS['y'])
    plt.legend()
    if save:
        save_dir = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['distortion_scan'], result.result_id)
        if not save_dir.is_dir():
            Path.mkdir(save_dir)
        title = str(result)
        if mpc:
            title = title + '_mpc'
        title = title + '.png'
        plt.savefig(Path(save_dir, title))
    plt.show()

    return chance_perf, distortion_values, accuracies


if __name__ == '__main__':

    # _result_id = '0004_rlt_0000_mdl_best_loss_0005_tst_b_scan_v2_blur'
    # _result_id = '0003_rlt_0000_mdl_best_loss_0003_tst_r_scan_v2_res'
    # _result_id = '0009_rlt_0000_mdl_best_loss_0007_tst_n_scan_v3_noise'

    _result_id = '0000_rlt_resnet18_pretrained_0000_tst_r_scan_pl_res'

    _distortion_id = 'res'
    _identifier = 'r_scan'
    _result, __ = get_model_distortion_performance_result(result_id=_result_id, identifier=_identifier,
                                                          distortion_ids=(_distortion_id,), make_dir=False)

    chance_perf_distortion(_result, _distortion_id, save=True, mpc=True)

