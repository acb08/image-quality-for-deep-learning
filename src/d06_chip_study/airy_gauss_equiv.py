import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, FIRST_DISC_ENCIRCLED_ENERGY


def integrate(x, y):

    total = 0
    integral = [total]
    for i, y_val in enumerate(y[:-1]):
        dx = x[i + 1] - x[i]
        area = (y[i] + 0.5 * (y[i + 1] - y[i])) * dx
        total += area
        integral.append(total)

    return np.asarray(integral)


def get_equivalent_gaussian_std(target_encircled_energy=None, save_plot=False):

    if not target_encircled_energy:
        target_encircled_energy = FIRST_DISC_ENCIRCLED_ENERGY

    r = np.linspace(0, 5, num=500)
    y = 1 / (2 * np.pi) * np.exp(-0.5 * r**2)
    energy = 2 * np.pi * r * y
    encircled_energy = integrate(r, energy)

    left_idx = np.where(encircled_energy <= target_encircled_energy)[0][-1]
    right_idx = np.where(encircled_energy >= target_encircled_energy)[0][0]
    r_equiv = (r[left_idx] + r[right_idx]) / 2 # approx, but ok for this application

    airy_equiv_line = target_encircled_energy * np.ones_like(r)

    airy_radius_plot = np.linspace(0, target_encircled_energy, num=50)
    airy_radius_plot_axis = r_equiv * np.ones_like(airy_radius_plot)

    plt.figure()
    plt.plot(r, y, label='2d Gaussian')
    plt.plot(r, y / np.max(y), label='scaled Gaussian PSF')
    plt.plot(r, encircled_energy, label='encircled energy')
    plt.plot(r, airy_equiv_line, label='airy encircled energy',
             linestyle='dashed')
    plt.plot(airy_radius_plot_axis, airy_radius_plot, label='airy radius',
             linestyle='dotted')
    plt.xlabel(r'psf radius (multiples of $\sigma_{Gaussian}$)')
    plt.ylabel('dimensionless')
    plt.legend(loc='lower right')
    if save_plot:
        save_dir = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['mtf_study'])
        if not save_dir.is_dir():
            Path.mkdir(save_dir)
        plt.savefig(Path(save_dir, 'gauss_equiv.png'))
    plt.show()

    return r_equiv


if __name__ == '__main__':

    equiv_gauss_std = get_equivalent_gaussian_std(save_plot=True)
    print(f'Equivalent Gaussian std: {equiv_gauss_std}')
