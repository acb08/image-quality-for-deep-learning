"""
Function for fits to be used when there is not a linear transform to enable fitting by singular value decomposition
"""
import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


def make_sample_data(params, x, sigma=0.1):
    y = giqe5_deriv(params, x)
    return y, y + np.random.randn(len(y)) * sigma


def giqe5_deriv(params, distortion_vector):

    c0, c1, c2, c3, c4, c5 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(blur) + c4 * np.log10(blur) \
        + c5 * noise

    return y


def _giqe5_deriv_residuals(params, y, distortion_vector):
    err = np.ravel(y) - giqe5_deriv(params, distortion_vector)
    return err


def power_law(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * res ** c2 + c3 * blur ** c4 + c5 * noise ** c6

    return y


def _power_law_residuals(params, y, distortion_vector):
    err = np.ravel(y) - power_law(params, distortion_vector)
    return err


def rer_0(params, distortion_vector):
    """
    If native blur is constant, c0 and c1 are redundant, but this function is designed to be extensible to
    true 2d distortion vectors in which native blur may vary
    """
    c0, c1, c2, c3 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = c0 - c1 * native_blur + c2 * np.exp(c3 * blur)

    return y


def _rer_0_residuals(params, y, distortion_vector):
    err = np.ravel(y) - rer_0(params, distortion_vector)
    return err


def rer_1(params, distortion_vector):
    """
    If native blur is constant, c0 and c1 are redundant, but this function is designed to be extensible to
    true 2d distortion vectors in which native blur may vary
    """
    c0, c1 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = c0 * np.exp(c1 * native_blur) * np.exp(c1 * blur)

    return y


def _rer_1_residuals(params, y, distortion_vector):
    err = np.ravel(y) - rer_1(params, distortion_vector)
    return err


def fit(x, y, distortion_ids=('res', 'blur', 'noise'), fit_key='giqe5_deriv'):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise Exception('distortion_ids must == (res, blur, noise)')

    residuals, initial_params = _leastsq_inputs[fit_key]
    w = leastsq(residuals, initial_params, args=(y, x))[0]

    return w


def apply_fit(w, x, fit_key):
    fit_function = _fit_functions[fit_key]
    return fit_function(w, x)


_leastsq_inputs = {
    'giqe5_deriv': (_giqe5_deriv_residuals, (0.5, 0.3, 0.3, -1, 0.3, -0.14)),
    'power_law': (_power_law_residuals, (0.5, 0.5, 1, -0.1, 1, -0.05, 1)),

    'rer_0': (_rer_0_residuals, (0.9, 0.25, 1, -1)),
    'rer_1': (_rer_1_residuals, (0.9, -1))
}

_fit_functions = {
    'giqe5_deriv': giqe5_deriv,
    'power_law': power_law,

    'rer_0': rer_0,
    'rer_1': rer_1
}


if __name__ == '__main__':

    _params = (0.5, 0.3, 0.3, -1.5, 0.3, -0.14)
    _num_pts = 13000

    _r = np.random.rand(_num_pts) * 0.8 + 0.2
    _sigma_b = np.random.rand(_num_pts) * 4.9 + 0.1
    _lambda_p = np.random.randint(0, 25, _num_pts) * 2

    _x0 = np.stack([_r, _sigma_b, _lambda_p], axis=1)
    print(np.shape(_x0))
    _truth, _data = make_sample_data(_params, _x0)

    _p_fit = fit(_x0, _data)
    # _p0 = np.array([8, -2, -3, -1, 1, -3])
    # _p_fit = leastsq(_giqe5_deriv_residuals, _p0, args=(_data, _x0))[0]

    _y_fit = giqe5_deriv(_p_fit, _x0)

    plt.figure()
    plt.scatter(np.arange(len(_data)), _data)
    plt.plot(_y_fit)
    plt.plot(_truth)
    plt.show()

    print(_p_fit)
