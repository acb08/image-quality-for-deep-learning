import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
import json

# def get_chips(images, v_start=10, h_start=10, n_v=25, n_h=25):
#     chips = {}
#     for key, image in images.items():
#         chip = image[v_start:v_start + n_v, h_start: h_start + n_h]
#         chips[key] = chip / 255
#     return chips
#
#
# def load_images():
#
#     img_1 = np.asarray(Image.open('edge1.png'))
#     img_2 = np.asarray(Image.open('edge2.png'))
#     img_3 = np.asarray(Image.open('edge3.png'))
#
#     images = {'no noise': img_1, 'low noise': img_2, 'high noise': img_3}
#
#     return images


def plot_rows(chip, interval=1):

    n_rows = np.shape(chip)[0]
    plt.figure()
    for i in range(n_rows):
        row = chip[i, :]
        if i % interval == 0:
            plt.plot(row)
    plt.show()


def get_lsf(esf, kernel=np.asarray([0.5, -0.5])):

    lsf = np.convolve(esf, kernel) * np.hamming(len(esf)+1)
    lsf = lsf[1:-1]
    lsf = lsf / np.sum(lsf)

    return lsf


def estimate_edge(row, kernel=np.asarray([-0.5, 0.5])):

    lsf = get_lsf(row, kernel=kernel)
    indices = np.arange(len(lsf))
    centroid = (np.sum(indices * lsf))/np.sum(lsf)

    return centroid


def fit_edge(rows, edge_method='standard', plot=True):

    n, m = np.shape(rows)
    centroids = np.zeros(n)
    row_indices = np.arange(n)
    for idx in row_indices:
        row = rows[idx, :]
        if edge_method == 'standard':
            centroid = estimate_edge(row)
        elif edge_method == 'tm':
            centroid = estimate_edge_tm(row)

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


def estimate_edge_tm(esf):

    m = len(esf)
    m1 = np.sum(esf) / m
    m2 = np.sum(esf**2) / m
    m3 = np.sum(esf**3) / m

    sigma = np.sqrt(m2 - m1**2)
    s = (m3 + 2 * m1**3 - 3 * m1 * m2) / sigma**3
    p = (1 + s * np.sqrt(1 / (4 + s**2))) / 2

    return p * m


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

    return x_oversampled, esf_estimate


def get_mtf(lsf):
    otf = np.fft.fft(lsf)
    mtf = np.abs(otf)
    mtf = mtf / np.max(mtf)
    return mtf


def estimate_mtf(chip, edge_method='standard'):

    centroids, fit = fit_edge(chip, edge_method=edge_method, plot=False)
    x_oversampled, esf = oversampled_esf(chip, fit)
    lsf = get_lsf(esf)
    mtf = get_mtf(lsf)

    return mtf, esf


def measure_mtf_lsf(dataset, directory):

    chip_data = dataset['chips']
    chip_dir = Path(directory, REL_PATHS['edge_chips'])
    output_dir = Path(directory, REL_PATHS['mtf'])
    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    mtf_lsf_data = {}

    for chip_name in chip_data.keys():

        chip = get_chip(chip_dir, chip_name)
        mtf_standard, esf = estimate_mtf(chip, edge_method='standard')
        mtf_lsf_data[str(chip_name)] = {
            'mtf': [float(val) for val in mtf_standard],
            'esf': [float(val) for val in esf]
        }
        freq_axis = get_freq_axis(mtf_standard)
        n_plot = int((len(freq_axis) / 2))
        n_plot_nyquist = int(n_plot / 2)

        plt.figure()
        plt.plot(freq_axis[:n_plot], mtf_standard[:n_plot], label='standard')
        plt.xlabel('spatial frequency [cycles / pixel]')
        plt.ylabel('MTF')
        plt.legend()
        plt.savefig(Path(output_dir, chip_name))
        plt.show()

        plt.figure()
        plt.plot(freq_axis[:n_plot_nyquist], mtf_standard[:n_plot_nyquist], label='standard')
        plt.xlabel('spatial frequency [cycles / pixel]')
        plt.ylabel('MTF')
        plt.legend()
        plt.savefig(Path(output_dir, f'nyquist_{chip_name}'))
        plt.show()

    with open(Path(directory, 'mtf_lsf.json'), 'w') as file:
        json.dump(mtf_lsf_data, file)


def get_freq_axis(mtf):

    f_max = 2
    f = np.linspace(0, f_max, num=len(mtf))
    return f


def load_dataset(directory_key):

    directory = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['mtf_study'], directory_key)
    with open(Path(directory, STANDARD_DATASET_FILENAME), 'r') as file:
        dataset = json.load(file)

    return directory, dataset


def get_chip(directory, name):
    return np.asarray(Image.open(Path(directory, name)))


if __name__ == '__main__':

    _directory_key = '0008'
    _directory, _dataset = load_dataset(_directory_key)

    measure_mtf_lsf(_dataset, _directory)
