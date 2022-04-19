import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
from src.d06_chip_study.measure_mtf import get_freq_axis, normed_circ_ap_mtf, load_dataset
from src.d06_chip_study.mtf_study_defitions import parent_names
import json


def q2_mtf_compare(mtf, oversample_factor=4):

    f_oversample = get_freq_axis(mtf, oversample_factor=oversample_factor)
    num_freqs_oversampled = len(f_oversample)

    f_cut = f_oversample[int(num_freqs_oversampled / oversample_factor)]
    idx_nyquist = int(num_freqs_oversampled / oversample_factor)
    idx_2x_nyquist = 2 * idx_nyquist
    f = f_oversample[:idx_2x_nyquist]

    q2_mtf = normed_circ_ap_mtf(f, f_cut=f_cut)
    plt.figure()
    plt.plot(f, mtf[:idx_2x_nyquist], label='measured')
    plt.plot(f, q2_mtf, label='Q=2 circular aperture')
    plt.legend()
    plt.show()


def load_measured_mtf_lsf(directory):
    with open(Path(directory, 'mtf_lsf.json'), 'r') as file:
        data = json.load(file)
    return data


def scale_freq_axis(f, resample_factor):
    return resample_factor * f


def get_nyquist(f_oversampled, oversample_factor):

    num_freqs_oversampled = len(f_oversampled)
    idx_nyquist = int(num_freqs_oversampled / oversample_factor)
    f_nyquist = f_oversampled[idx_nyquist]

    return idx_nyquist, f_nyquist


def truncate(f, mtf, f_cut=0.5):

    f_keep = f[np.where(f <= f_cut)]
    mtf_keep = mtf[np.where(f <= f_cut)]

    return f_keep, mtf_keep


def mtf_evolution_plot(chip_id, dataset, mtf_lsf_data, directory=None):

    if directory:
        save_dir = Path(directory, REL_PATHS['mtf'])
        save_name = f"{chip_id.split('.')[0]}_mtf_evolution.png"

    chip_mtf = np.asarray(mtf_lsf_data[chip_id]['mtf'])
    oversample_factor = mtf_lsf_data[chip_id]['oversample_factor']

    parent_ids = dataset['chips'][chip_id]['parents']
    parent_mtfs = []
    parent_freq_axes = []
    for parent_id in parent_ids:
        parent_mtf = np.asarray(mtf_lsf_data[parent_id]['mtf'])
        parent_freq_axis = get_freq_axis(parent_mtf)
        parent_freq_axis = scale_freq_axis(parent_freq_axis, len(parent_freq_axis) / len(chip_mtf))
        parent_freq_axis, parent_mtf = truncate(parent_freq_axis, parent_mtf)
        parent_freq_axes.append(parent_freq_axis)
        parent_mtfs.append(parent_mtf)

    freq_axis = get_freq_axis(chip_mtf, oversample_factor=oversample_factor)
    freq_axis, chip_mtf = truncate(freq_axis, chip_mtf)

    plt.figure()
    for i, parent_mtf in enumerate(parent_mtfs):
        plt.plot(parent_freq_axes[i], parent_mtfs[i], label=parent_names[i])
    plt.plot(freq_axis, chip_mtf, label='chip')
    plt.xlabel('cycles per native pixel')
    plt.ylabel('mtf')
    plt.legend()
    if directory:
        plt.savefig(Path(save_dir, save_name))
    plt.show()


if __name__ == '__main__':

    _directory_key = '0018'
    _directory, _dataset = load_dataset(_directory_key)
    _mtf_lsf_data = load_measured_mtf_lsf(_directory)
    for _chip_id in _dataset['chips'].keys():
        mtf_evolution_plot(_chip_id, _dataset, _mtf_lsf_data, directory=_directory)

