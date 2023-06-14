import numpy as np
import matplotlib.pyplot as plt


def relative_camera_equation(fn0, sigma_blur, r):
    f_num_scale_factor = (sigma_blur / r) ** 2
    return (1 + 4 * fn0 ** 2) / (1 + 4 * fn0 ** 2 * f_num_scale_factor)


if __name__ == '__main__':

    _sigma_vals = np.arange(1, 6)
    _r = np.linspace(0.1, 1, num=10)
    _fn0_vals = np.arange(4) + 0.5

    for _sigma_val in _sigma_vals:
        for _fn0_val in _fn0_vals:
            relative_irradiance = relative_camera_equation(_fn0_val, _sigma_val, _r)

