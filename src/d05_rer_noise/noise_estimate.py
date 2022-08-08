"""
Makes a very rough estimate of the noise content in a typical 8-bit image with 50% saturation in its RGB format and
after conversion to grayscale
"""

import numpy as np
import argparse
from src.d00_utils import functions, definitions
from pathlib import Path
import matplotlib.pyplot as plt

RNG = np.random.default_rng()


def pre_sample_electrons(image_size, saturation_frac, well_depth, dark_electrons, read_noise):
    """
    Generates an array representative of pre-digitization electron counts in a detector array with constant signal
    (defined in terms of saturation fraction), a known expected dark current count, and read noise defined by a
    Gaussian standard deviation. Output incorporates signal shot noise.
    """

    electrons = well_depth * saturation_frac * np.ones((image_size, image_size, 3))  # expected signal electrons
    electrons = add_dark_electrons(electrons, dark_electrons)  # expected signal plus dark current electrons
    electrons = incorporate_shot_noise(electrons)  # Poisson distribution covering shot noise in signal & dark electrons
    electrons = add_read_noise(electrons, read_noise, well_depth=well_depth)  # Gaussian noise (std=read_noise, mean=0)

    return electrons


def electrons_to_counts(electrons, well_depth, bit_depth=8):

    scaled_values = electrons / well_depth
    max_counts = 2 ** bit_depth
    counts = max_counts * scaled_values
    counts = np.asarray(counts, dtype=np.int64)

    return counts


def rgb_to_pan(image):
    return np.mean(image, axis=2)


def estimate_noise(counts):
    return np.std(counts)


def incorporate_shot_noise(expected_electrons):
    return RNG.poisson(expected_electrons)


def add_read_noise(image_electrons, read_noise, well_depth):
    """
    Incorporates Gaussian read noise, capped at well depth
    """
    read_electrons = RNG.normal(read_noise, size=np.shape(image_electrons))
    image_electrons = image_electrons + read_electrons
    image_electrons = np.clip(image_electrons, 0, well_depth)
    return image_electrons + read_electrons


def add_dark_electrons(image_electrons, dark_count):
    """
    Simple addition of expected dark electrons, with shot noise incorporated after signal and dark electrons
    accumulated
    """
    expected_dark_electrons = dark_count * np.ones_like(image_electrons)  # shot noise added later
    return image_electrons + expected_dark_electrons


def make_image(image_size, saturation_frac, well_depth, dark_electrons, read_noise, bit_depth):
    """
    Returns an image array for constant signal at saturation_frac, converted to an integer, where electrons per count
    is given by well_depth / 2 ** bit_depth
    """
    electrons = pre_sample_electrons(image_size=image_size, saturation_frac=saturation_frac, well_depth=well_depth,
                                     dark_electrons=dark_electrons, read_noise=read_noise)
    counts = electrons_to_counts(electrons, well_depth=well_depth, bit_depth=bit_depth)
    return counts


def estimate_snr(image_size, saturation, well_depth, dark_electrons, read_noise, bit_depth):
    """
    Estimates SNR of an RGB image and it's grayscale counterpoint by comparing a light patch and dark patch, with light
    patch signal specified by saturation and dark patch signal set to zero (before dark current and readout noise).
    Signal is defined here as the mean difference between the light and dark patch, and noise is defined as the standard
    deviation of the difference.
    """

    light_patch = make_image(image_size, saturation_frac=saturation, well_depth=well_depth,
                             dark_electrons=dark_electrons, read_noise=read_noise, bit_depth=bit_depth)
    dark_patch = make_image(image_size, saturation_frac=0, well_depth=well_depth, dark_electrons=dark_electrons,
                            read_noise=read_noise, bit_depth=bit_depth)

    assert np.min(light_patch) >= 0
    assert np.min(dark_patch) >= 0

    diff = light_patch - dark_patch
    noise = estimate_noise(diff)
    snr = np.mean(diff) / noise

    pan_light_path = np.mean(light_patch, axis=2)
    pan_dark_path = np.mean(dark_patch, axis=2)
    pan_diff = pan_light_path - pan_dark_path
    pan_noise = estimate_noise(pan_diff)
    pan_snr = np.mean(pan_diff) / pan_noise

    return snr, noise, pan_snr, pan_noise


def estimate_basic_pan_image_noise(image_size, saturation, well_depth, dark_electrons, read_noise, bit_depth):
    image = make_image(image_size=image_size, saturation_frac=saturation, well_depth=well_depth,
                       dark_electrons=dark_electrons, read_noise=read_noise, bit_depth=bit_depth)
    image = rgb_to_pan(image)
    noise = estimate_noise(image)
    return noise


def noise_sweep(image_size, saturation_frac, well_depth, dark_electron_values, read_noise_values, bit_depth=8):

    n_dark, n_read = len(dark_electron_values), len(read_noise_values)

    rgb_snr = np.zeros((n_dark, n_read))
    rgb_noise = np.zeros((n_dark, n_read))
    pan_snr = np.zeros((n_dark, n_read))
    pan_noise = np.zeros((n_dark, n_read))

    for i, dark_electrons in enumerate(dark_electron_values):
        for j, read_noise in enumerate(read_noise_values):
            rgb_snr_val, rgb_noise_val, pan_snr_val, pan_noise_val = estimate_snr(image_size=image_size,
                                                                                  saturation=saturation_frac,
                                                                                  well_depth=well_depth,
                                                                                  dark_electrons=dark_electrons,
                                                                                  read_noise=read_noise,
                                                                                  bit_depth=bit_depth)
            rgb_snr[i, j] = rgb_snr_val
            rgb_noise[i, j] = rgb_noise_val
            pan_snr[i, j] = pan_snr_val
            pan_noise[i, j] = pan_noise_val

    return rgb_snr, rgb_noise, pan_snr, pan_noise


def log_noise_matrices_txt(dark_current, read_noise, rgb_snr, rgb_noise, pan_snr, pan_noise, output_dir=None):

    if output_dir:
        output_path = Path(output_dir, 'noise_results.txt')
        output_file = open(output_path, 'w')
    else:
        output_file = None

    print('dark electron value:', file=output_file)
    print(dark_current, '\n', file=output_file)
    print('read noise values: ', file=output_file)
    print(read_noise, '\n', file=output_file)

    print('rgb snr: ', file=output_file)
    print(rgb_snr, '\n', file=output_file)
    print('rgb noise: ', file=output_file)
    print(rgb_noise, '\n', file=output_file)

    print('pan snr: ', file=output_file)
    print(pan_snr, '\n', file=output_file)
    print('pan noise: ', file=output_file)
    print(pan_noise, '\n', file=output_file)

    if output_file:
        output_file.close()


def main(config):

    image_size = config['image_size']
    saturation_frac = config['saturation_frac']
    well_depth = config['well_depth']
    dark_electron_values = config['dark_electron_values']
    read_noise_values = config['read_noise_values']
    bit_depth = config['bit_depth']

    parent_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['noise_study'])
    if not parent_dir.is_dir():
        Path.mkdir(parent_dir)

    key, __, __ = functions.key_from_dir(parent_dir)
    output_dir = Path(parent_dir, key)
    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    functions.log_config(output_dir, config)

    rgb_snr, rgb_noise, pan_snr, pan_noise = noise_sweep(image_size,
                                                         saturation_frac,
                                                         well_depth,
                                                         dark_electron_values,
                                                         read_noise_values,
                                                         bit_depth=bit_depth)

    log_noise_matrices_txt(dark_current=dark_electron_values,
                           read_noise=read_noise_values,
                           rgb_snr=rgb_snr,
                           rgb_noise=rgb_noise,
                           pan_snr=pan_snr,
                           pan_noise=pan_noise,
                           output_dir=output_dir)

    noise_plot(rgb_snr.T, dark_electron_values, xlabel='dark electrons', data_labels_vector=read_noise_values,
               data_label_string='read electrons', ylabel='snr', output_dir=output_dir, filename='rgb_snr.png')

    noise_plot(pan_snr.T, dark_electron_values, xlabel='dark electrons', data_labels_vector=read_noise_values,
               data_label_string='read electrons', ylabel='snr', output_dir=output_dir, filename='pan_snr.png')

    noise_plot(pan_noise.T, dark_electron_values, xlabel='dark electrons', data_labels_vector=read_noise_values,
               data_label_string='read electrons', ylabel='noise', output_dir=output_dir, filename='pan_noise.png')


def noise_plot(data_array, x, xlabel, data_labels_vector, data_label_string=None, ylabel=None, output_dir=None,
               filename=None):
    """
    Makes 1d plots using the rows in data array as the dependent variable.

    :param data_array: n-by-m array
    :param x: array of length m with the independent variable to plotted
    :param xlabel: string
    :param data_labels_vector: array of length n with labels (presumably numerical) corresponding to each
    row of the data vector
    :param data_label_string: string to be used in labeling data (essentially an identifier for the values in
    data_labels_vector)
    :param ylabel: string
    :param output_dir: string/path
    :param filename: string
    """

    plt.figure()
    for i, row in enumerate(data_array[:]):
        if data_label_string:
            label = f'{data_labels_vector[i]} {data_label_string}'
        else:
            label = str(data_labels_vector[i])
        plt.plot(x, row, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if output_dir:
        plt.savefig(Path(output_dir, filename))
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='noise_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'noise_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = functions.get_config(args_passed)

    main(run_config)
