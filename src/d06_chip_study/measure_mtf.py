import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
import json


def plot_rows(chip, interval=1):

    n_rows = np.shape(chip)[0]
    plt.figure()
    for i in range(n_rows):
        row = chip[i, :]
        if i % interval == 0:
            plt.plot(row)
    plt.show()


def pad_signal(x, num):

    x_new = np.zeros(len(x) + 2 * num)
    x_new[:num] = x[0]
    x_new[num:-num] = x
    x_new[-num:] = x[-1]

    return x_new


def get_lsf(esf, kernel=np.asarray([0.5, 0, -0.5]), pad=False):

    if pad:
        num_pad = int(np.floor(len(kernel) / 2))
        esf = pad_signal(esf, num_pad)

    lsf = np.convolve(esf, kernel, mode='valid')
    lsf = lsf * np.hamming(len(lsf))
    lsf = lsf / np.sum(lsf)

    return lsf


def estimate_edge(row, kernel=np.asarray([-0.5, 0.5])):

    lsf = get_lsf(row, kernel=kernel)
    indices = np.arange(len(lsf))
    centroid = (np.sum(indices * lsf))/np.sum(lsf)

    return centroid


def fit_edge(rows, plot=True):

    n, m = np.shape(rows)
    centroids = np.zeros(n)
    row_indices = np.arange(n)
    for idx in row_indices:
        row = rows[idx, :]
        centroid = estimate_edge(row)

        centroids[idx] = centroid

    fit = stats.linregress(row_indices, centroids)
    slope, offset = fit[0], fit[1]

    if plot:

        best_fit_line = slope * row_indices + offset
        plt.figure()
        plt.scatter(row_indices, centroids, marker='+', s=30, color='r')
        plt.plot(row_indices, best_fit_line, color='b')
        plt.xlabel('row index')
        plt.ylabel('edge location')
        plt.show()

    return centroids, fit


def oversampled_esf(rows, fit, oversample_factor=4, plot=True):

    n, m = np.shape(rows)
    x_coarse = np.arange(m)
    row_indices = np.arange(n)
    x_shifted = np.zeros((n, m))

    slope, offset = fit[0], fit[1]

    for idx in row_indices:
        row_shift = -slope * idx + offset
        x_scaled_shifted = oversample_factor * (x_coarse + row_shift)
        x_scaled_quantized = np.rint(x_scaled_shifted)
        x_shifted[idx] = x_scaled_quantized / oversample_factor

    x_oversampled = np.unique(x_shifted)

    esf_estimate = np.zeros_like(x_oversampled)
    for idx, x_val in enumerate(x_oversampled):
        indices = np.where(x_shifted == x_val)
        esf_mean = np.mean(rows[indices])
        esf_estimate[idx] = esf_mean

    if plot:

        plt.figure()
        for i in range(n):
            plt.scatter(x_shifted[i], rows[i])
        plt.plot(x_oversampled, esf_estimate)
        plt.title('shifted')
        plt.show()

    target_length = oversample_factor * m
    x_oversampled, esf_estimate = fix_sample_length(x_oversampled, esf_estimate,
                                                    target_length)

    return x_oversampled, esf_estimate, oversample_factor


def fix_sample_length(x_oversampled, esf_estimate, target_length):

    n_start = len(x_oversampled)
    if n_start > target_length:
        x_oversampled = x_oversampled[-target_length:]
        esf_estimate = esf_estimate[-target_length:]

    if n_start < target_length:

        print('warning, fewer edge samples than expected')
        x_ext = np.zeros(target_length)
        x_ext[:len(x_oversampled)] = x_oversampled
        delta_x = x_oversampled[1] - x_oversampled[0]
        last_x = x_oversampled[-1]
        num_extend = target_length - len(x_oversampled)
        extension = [(last_x + delta_x * (i + 1)) for i in range(num_extend)]
        x_ext[-num_extend:] = extension

        esf_ext = np.zeros_like(x_ext)
        esf_ext[-num_extend:] = esf_estimate[-1]

        return x_ext, esf_ext

    return x_oversampled, esf_estimate


def get_mtf(lsf):
    otf = np.fft.fft(lsf)
    mtf = np.abs(otf)
    mtf = mtf / np.max(mtf)
    return mtf


def estimate_mtf(chip):

    centroids, fit = fit_edge(chip, plot=False)
    x_oversampled, esf, oversample_factor = oversampled_esf(chip, fit)
    lsf = get_lsf(esf, pad=True)
    mtf = get_mtf(lsf)

    print('esf:', len(esf), ' lsf:', len(lsf), ' mtf: ', len(mtf))

    return mtf, esf, oversample_factor


def measure_mtf_lsf(dataset, directory):

    chip_data = dataset['chips']
    edge_names = dataset['edges']
    chip_dir = Path(directory, REL_PATHS['edge_chips'])
    edge_dir = Path(directory, REL_PATHS['edges'])
    output_dir = Path(directory, REL_PATHS['mtf'])
    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    mtf_lsf_data = {}

    for edge_name in edge_names:
        edge = get_image_array(edge_dir, edge_name)
        mtf, esf, oversample_factor = estimate_mtf(edge)
        mtf_lsf_data[edge_name] = {
            'mtf': list(mtf),
            'esf': list(esf),
            'oversample_factor': oversample_factor
        }
        plot_mtf(mtf, output_dir=output_dir, chip_name=edge_name)

    for chip_name in chip_data.keys():

        chip = get_image_array(chip_dir, chip_name)
        mtf, esf, oversample_factor = estimate_mtf(chip)
        mtf_lsf_data[str(chip_name)] = {
            'mtf': [float(val) for val in mtf],
            'esf': [float(val) for val in esf],
            'oversample_factor': oversample_factor
        }
        plot_mtf(mtf, output_dir=output_dir, chip_name=chip_name)

    with open(Path(directory, 'mtf_lsf.json'), 'w') as file:
        json.dump(mtf_lsf_data, file)

    return mtf_lsf_data


def get_freq_axis(mtf, oversample_factor=4):
    f_max = oversample_factor / 2
    f = np.linspace(0, f_max, num=len(mtf))
    return f


def plot_mtf(mtf, output_dir=None, chip_name=None, f_max=None):

    freq_axis = get_freq_axis(mtf)
    n_plot = int((len(freq_axis) / 2))
    n_plot_nyquist = int(n_plot / 2)

    plt.figure()
    plt.plot(freq_axis[:n_plot], mtf[:n_plot], label='standard')
    plt.xlabel('spatial frequency [cycles / pixel]')
    plt.ylabel('MTF')
    plt.legend()
    if output_dir and chip_name:
        plt.savefig(Path(output_dir, chip_name))
    plt.show()

    plt.figure()
    plt.plot(freq_axis[:n_plot_nyquist], mtf[:n_plot_nyquist], label='standard')
    plt.xlabel('spatial frequency [cycles / pixel]')
    plt.ylabel('MTF')
    plt.legend()
    if output_dir and chip_name:
        plt.savefig(Path(output_dir, f'nyquist_{chip_name}'))
    plt.show()

    pass


def load_dataset(directory_key):

    directory = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['mtf_study'], directory_key)
    with open(Path(directory, STANDARD_DATASET_FILENAME), 'r') as file:
        dataset = json.load(file)

    return directory, dataset


def get_image_array(directory, name):
    return np.asarray(Image.open(Path(directory, name)))


def normed_circ_ap_mtf(f, f_cut=None):
    """
    Calculates the diffraction mtf over frequency axis f for a circular aperture with optical cutoff frequency f_cut.
    """
    if not f_cut:
        f_cut = max(f)
    f = f / f_cut
    mtf = np.zeros_like(f)
    f_sub_cut = f[np.where(f <= 1)]
    mtf_sub_cut = 2 / np.pi * (np.arccos(f_sub_cut) - f_sub_cut * np.sqrt(1 - f_sub_cut**2))
    samples_sub_cut = len(mtf_sub_cut)
    mtf[:samples_sub_cut] = mtf_sub_cut

    return mtf


if __name__ == '__main__':

    _directory_key = '0018'
    _directory, _dataset = load_dataset(_directory_key)

    measure_mtf_lsf(_dataset, _directory)

