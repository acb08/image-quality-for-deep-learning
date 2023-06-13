"""
An exploration on how to simulate Poisson noise in re-processed images.
"""

import numpy as np
from src.pre_processing.distortion_tools import image_to_electrons, electrons_to_image, \
    apply_partial_poisson_distribution
from src.pre_processing.distortions import get_kernel_size
from torchvision import transforms
import matplotlib.pyplot as plt
from src.analysis.plot import plot_2d
from scipy import interpolate
from src.utils.definitions import ROOT_DIR, REL_PATHS, BASELINE_HIGH_SIGNAL_WELL_DEPTH
from pathlib import Path
from src.utils.functions import id_from_tags
import matplotlib

matplotlib.use('TkAgg')


def apply_blur(img, std):
    kernel_size = get_kernel_size(std)
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img)


def initial_electrons_and_image(fill_fraction=0.5, shape=(128, 128), poisson_noise=True):

    assert 1 >= fill_fraction >= 0

    electrons = fill_fraction * BASELINE_HIGH_SIGNAL_WELL_DEPTH * np.ones(shape, dtype=np.float32)
    if poisson_noise:
        electrons = apply_partial_poisson_distribution(electrons)
    image = electrons_to_image(electrons)

    return electrons, image


def get_root(x, y):
    interp = interpolate.UnivariateSpline(x, y, s=0)
    root = interp.roots()
    return root


def interpolate_constant_noise_dc_fractions(delta_noise_electrons, blur_stds, dc_fractions):

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


def main(blur_stds,
         dc_fractions,
         well_fill_fraction,
         shape,
         manual_dir_name=None,
         make_2d_plots=True,
         make_1d_plot=True):

    if manual_dir_name is None:
        tags = []
        dir_name, rel_dir = id_from_tags('poisson_sim', tags, return_dir=True)
        output_dir = Path(ROOT_DIR, rel_dir)

    else:
        output_dir = Path(ROOT_DIR, REL_PATHS['poisson_sim'], manual_dir_name)
    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    pre_image_electrons, img = initial_electrons_and_image(fill_fraction=well_fill_fraction, shape=shape)
    original_noise_dn = np.std(img)
    original_img_snr = np.mean(img) / original_noise_dn

    pre_image_electron_noise = np.std(pre_image_electrons)
    pre_image_snr = np.mean(pre_image_electrons) / pre_image_electron_noise

    estimated_original_electrons = image_to_electrons(img)
    original_noise_electrons_estimated = np.std(estimated_original_electrons)
    original_snr_electrons_estimated = np.mean(estimated_original_electrons) / original_noise_electrons_estimated

    final_noise_electrons = np.zeros((len(blur_stds), len(dc_fractions)))
    final_noise_dn = np.zeros_like(final_noise_electrons)

    post_blur_noise_electron_values = np.zeros((len(blur_stds)))
    post_blur_noise_dn_values_0 = np.zeros_like(post_blur_noise_electron_values)
    post_blur_noise_dn_values_1 = np.zeros_like(post_blur_noise_electron_values)

    for i, blur_std in enumerate(blur_stds):

        blurred_img = apply_blur(img=img, std=blur_std)
        noise_dn_0 = np.std(blurred_img)
        post_blur_noise_dn_values_0[i] = noise_dn_0

        electrons = image_to_electrons(blurred_img)

        electron_noise = np.std(electrons)
        post_blur_noise_electron_values[i] = electron_noise
        blurred_img = electrons_to_image(electrons)
        noise_dn_1 = np.std(blurred_img)
        post_blur_noise_dn_values_1[i] = noise_dn_1

        for j, dc_frac in enumerate(dc_fractions):
            shot_noised_signal = apply_partial_poisson_distribution(electrons,
                                                                    dc_fraction=dc_frac)

            noise_electrons = np.std(shot_noised_signal)
            final_noise_electrons[i, j] = noise_electrons

            post_noise_image = electrons_to_image(shot_noised_signal)
            final_noise_dn[i, j] = np.std(post_noise_image)

    delta_noise_electrons = final_noise_electrons - original_noise_electrons_estimated
    blur_stds_with_roots, best_dc_fractions = interpolate_constant_noise_dc_fractions(delta_noise_electrons,
                                                                                      blur_stds, dc_fractions)
    delta_noise_dn = final_noise_dn - original_noise_dn
    blur_stds_with_roots_dn, best_dc_fractions_dn = interpolate_constant_noise_dc_fractions(delta_noise_dn,
                                                                                            blur_stds, dc_fractions)

    if make_1d_plot:

        plt.figure()
        plt.plot(blur_stds_with_roots, best_dc_fractions)
        plt.xlabel(r'$\sigma$-blur')
        plt.ylabel('DC electron fraction')
        plt.savefig(Path(output_dir, 'interpolated_dc_fractions_electrons.png'))
        plt.show()

        plt.figure()
        plt.plot(blur_stds_with_roots_dn, best_dc_fractions_dn)
        plt.xlabel(r'$\sigma$-blur')
        plt.ylabel('DC electron fraction')
        plt.savefig(Path(output_dir, 'interpolated_dc_fractions_electrons_dn.png'))
        plt.show()

    if make_2d_plots:

        plot_2d(x_values=blur_stds,
                y_values=dc_fractions,
                accuracy_means=delta_noise_electrons,
                x_id='x',
                y_id='y',
                zlabel='delta noise (electrons)',
                result_identifier='delta_noise_electrons',
                axis_labels={'x': 'blur std', 'y': 'dc fraction'},
                az_el_combinations='mini',
                directory=output_dir,
                show_plots=True,
                perf_metric='std',
                z_limits=None,
                sub_dir_per_az_el=False
                )

        plot_2d(x_values=blur_stds,
                y_values=dc_fractions,
                accuracy_means=final_noise_dn - original_noise_dn,
                x_id='x',
                y_id='y',
                zlabel='delta noise (DN)',
                result_identifier='delta_noise_dn',
                axis_labels={'x': 'blur std', 'y': 'dc fraction'},
                az_el_combinations='mini',
                directory=output_dir,
                show_plots=True,
                perf_metric='std',
                z_limits=None,
                sub_dir_per_az_el=False
                )


if __name__ == '__main__':

    _blur_stds = (0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5)
    _dc_fractions = np.linspace(0, 0.9, num=15)
    _well_fill_fraction = 0.5
    _shape = (256, 256)
    _manual_name = '50-percent'

    main(blur_stds=_blur_stds,
         dc_fractions=_dc_fractions,
         well_fill_fraction=_well_fill_fraction,
         shape=_shape,
         make_1d_plot=True,
         make_2d_plots=True)
