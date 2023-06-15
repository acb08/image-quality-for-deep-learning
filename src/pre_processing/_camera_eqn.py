import numpy as np
import matplotlib.pyplot as plt
from src.utils.definitions import PSEUDO_SYSTEM_DISTORTION_RANGE


def sensor_radiance_ratio(baseline_f_number, sigma_blur=None, r=None, scale_factor=None):
    if scale_factor is None:
        scale_factor = sigma_blur * r
    scaled_f_number = scale_factor * baseline_f_number
    return (1 + 4 * baseline_f_number ** 2) / (1 + 4 * scaled_f_number ** 2)


if __name__ == '__main__':

    baseline_f_numbers = [3, 5, 7, 9]
    scale_factors = np.linspace(0.2, 1, num=21)
    abbrev_scale_factors = np.linspace(0.2, 5, num=7)

    plt.figure()
    for f_number in baseline_f_numbers:
        ratio = sensor_radiance_ratio(baseline_f_number=f_number,
                                      scale_factor=scale_factors)

        plt.plot(scale_factors, ratio, label=f'F0 = {round(f_number, 2)}')
        plt.xlabel(r'f-number scale factor ($r \times \sigma$)')
        plt.ylabel('radiance ratio')
    plt.legend()
    plt.show()

