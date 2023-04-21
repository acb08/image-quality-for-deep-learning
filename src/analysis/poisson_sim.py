"""
An exploration on how to simulate Poisson noise in re-processed images.
"""

import numpy as np
from src.pre_processing.distortions import get_kernel_size
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.analysis.plot import plot_2d
from scipy import interpolate

WELL_DEPTH = 10_000
N_BITS = 8


def image_to_electrons(image):

    # assert type(image) == Image.Image

    image = np.asarray(image, dtype=np.uint8)
    image = normalize(image)
    electrons = image * WELL_DEPTH
    electrons = np.asarray(electrons, dtype=np.int32)

    return electrons


def electrons_to_image(electrons):

    image = electrons / WELL_DEPTH
    image = convert_to_uint8(image)
    image = Image.fromarray(image)

    return image


def apply_partial_poisson_distribution(signal, dc_fraction=0):

    dc_component = dc_fraction * signal
    poisson_component = signal - dc_component

    return dc_component + np.random.poisson(poisson_component)


def apply_blur(img, std):
    kernel_size = get_kernel_size(std)
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img)


def initial_image(fill_fraction=0.5, shape=(64, 64), poisson_noise=True):

    assert 1 >= fill_fraction >= 0

    electrons = fill_fraction * WELL_DEPTH * np.ones(shape, dtype=np.float32)
    if poisson_noise:
        electrons = apply_partial_poisson_distribution(electrons)
    image = electrons_to_image(electrons)

    return image


def normalize(img):

    assert np.max(img) <= 255
    assert np.min(img) >= 0

    return img / (2 ** 8 - 1)


def convert_to_uint8(img):

    assert np.min(img) >= 0.0
    assert np.max(img) <= 1.0

    img = (2 ** 8 - 1) * img
    img = np.asarray(img, dtype=np.uint8)

    return img


def get_root(x, y):
    interp = interpolate.UnivariateSpline(x, y, s=0)
    root = interp.roots()
    return root


def find_ideal_dc_fraction(delta_noise_electrons, blur_stds, dc_fractions):

    n_stds, n_fractions = np.shape(delta_noise_electrons)
    assert (n_stds, n_fractions) == (len(blur_stds), len(dc_fractions))

    blur_stds_with_roots = []
    best_dc_fractions = []

    for blur_idx in range(n_stds):
        delta_noise_array = delta_noise_electrons[blur_idx, :]
        dc_frac_zero_change = get_root(dc_fractions, delta_noise_array)

        if len(dc_frac_zero_change) == 1:
            blur_stds_with_roots.append(blur_stds[blur_idx])
            best_dc_fractions.append(dc_frac_zero_change)

    return blur_stds_with_roots, best_dc_fractions


if __name__ == '__main__':

    _well_fill_fractions = np.linspace(0.1, 0.9, num=8)

    _stds = []

    for _frac in _well_fill_fractions:
        _img = initial_image(fill_fraction=_frac)
        _std = np.std(np.asarray(_img))
        _stds.append(_std)

    plt.figure()
    plt.plot(_well_fill_fractions, _stds)
    plt.xlabel('fill fraction')
    plt.ylabel('image std')
    plt.show()

    _blur_stds = (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5)
    _dc_fractions = np.linspace(0, 0.4, num=10)

    _well_fill_fraction = 0.5
    _shape = (256, 256)
    _img = initial_image(fill_fraction=_well_fill_fraction)
    _original_noise_dn = np.std(_img)
    _original_img_snr = np.mean(_img) / _original_noise_dn

    _original_electrons = image_to_electrons(_img)
    _original_noise_electrons = np.std(_original_electrons)
    _original_snr_electrons = np.mean(_original_electrons) / _original_noise_electrons

    print(_original_snr_electrons, _original_img_snr, _original_noise_electrons)

    _final_noise_electrons = np.zeros((len(_blur_stds), len(_dc_fractions)))
    _final_noise_dn = np.zeros_like(_final_noise_electrons)

    _post_blur_noise_electron_values = np.zeros((len(_blur_stds)))
    _post_blur_noise_dn_values_0 = np.zeros_like(_post_blur_noise_electron_values)
    _post_blur_noise_dn_values_1 = np.zeros_like(_post_blur_noise_electron_values)

    for i, _blur_std in enumerate(_blur_stds):

        _blurred_img = apply_blur(img=_img, std=_blur_std)
        _noise_dn_0 = np.std(_blurred_img)
        _post_blur_noise_dn_values_0[i] = _noise_dn_0

        _electrons = image_to_electrons(_blurred_img)

        _electron_noise = np.std(_electrons)
        _post_blur_noise_electron_values[i] = _electron_noise
        _blurred_img = electrons_to_image(_electrons)
        _noise_dn_1 = np.std(_blurred_img)

        for j, _dc_frac in enumerate(_dc_fractions):
            _shot_noised_signal = apply_partial_poisson_distribution(_electrons,
                                                                     dc_fraction=_dc_frac)

            _noise_electrons = np.std(_shot_noised_signal)
            _final_noise_electrons[i, j] = _noise_electrons

            _post_noise_image = electrons_to_image(_shot_noised_signal)
            _final_noise_dn[i, j] = np.std(_post_noise_image)

    _delta_noise_electrons = _final_noise_electrons - _original_noise_electrons
    _blur_stds_with_roots, _best_dc_fractions = find_ideal_dc_fraction(_delta_noise_electrons, _blur_stds, _dc_fractions)

    plt.figure(), plt.plot(_blur_stds_with_roots, _best_dc_fractions), plt.show()

    plot_2d(x_values=_blur_stds,
            y_values=_dc_fractions,
            accuracy_means=_final_noise_electrons - _original_noise_electrons,
            x_id='x',
            y_id='y',
            zlabel='delta noise (electrons)',
            result_identifier=None,
            axis_labels={'x': 'blur std', 'y': 'dc fraction'},
            az_el_combinations='mini',
            directory=None,
            show_plots=True,
            perf_metric='std',
            z_limits=None,
            sub_dir_per_az_el=False
            )

    plot_2d(x_values=_blur_stds,
            y_values=_dc_fractions,
            accuracy_means=_final_noise_dn - _original_noise_dn,
            x_id='x',
            y_id='y',
            zlabel='delta noise (DN)',
            result_identifier=None,
            axis_labels={'x': 'blur std', 'y': 'dc fraction'},
            az_el_combinations='mini',
            directory=None,
            show_plots=True,
            perf_metric='std',
            z_limits=None,
            sub_dir_per_az_el=False
            )




