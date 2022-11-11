"""
Function for fits to be used when there is not a linear transform to enable fitting by singular value decomposition
"""
import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt

from src.d04_analysis.fit_functions import giqe35_b0n0, fit_functions


class Fitter(object):

    def __init__(self, x, y, fit_function, initial_params):

        self.x = x
        self.y = y
        self.fit_function = fit_function
        self.initial_params = initial_params

    def residuals(self, params):
        return np.ravel(self.y) - np.ravel(self.fit_function(params, self.x))

    def fit(self):
        return leastsq(self.residuals, self.initial_params)[0]


def make_sample_data(params, x, sigma=0.1):
    y = giqe35_b0n0(params, x)
    return y, y + np.random.randn(len(y)) * sigma


def fit(x, y, distortion_ids=('res', 'blur', 'noise'), fit_key='giqe35_b0n0'):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise Exception('distortion_ids must == (res, blur, noise)')

    fit_function, initial_params = fit_functions[fit_key]
    w = Fitter(x, y, fit_function, initial_params).fit()

    return w


def apply_fit(w, x, fit_key):
    fit_function = fit_functions[fit_key][0]
    return fit_function(w, x)


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

    _y_fit = giqe35_b0n0(_p_fit, _x0)

    plt.figure()
    plt.scatter(np.arange(len(_data)), _data)
    plt.plot(_y_fit)
    plt.plot(_truth)
    plt.show()

    print(_p_fit)
