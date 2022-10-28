"""
Function for fits to be used when there is not a linear transform to enable fitting by singular value decomposition
"""
import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


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
    y = giqe5_deriv(params, x)
    return y, y + np.random.randn(len(y)) * sigma


def giqe5_deriv(params, distortion_vector):

    c0, c1, c2, c3, c4, c5 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(blur) + c4 * np.log10(blur) \
        + c5 * noise

    return y


def giqe5_deriv_2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    rer = c4 / (1 + c5 * blur)
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c6 * np.log10(rer) \
        + c7 * noise

    return y


def giqe5_deriv_3(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    rer = c4 / (1 + c5 * blur / res)
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c6 * np.log10(rer) \
        + c7 * noise

    return y


def giqe5_deriv_4(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    rer = 1 / np.sqrt(2 * np.pi * (c4 * res + blur ** 2))  # scale the native blur by res since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe5_deriv_5(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = noise + noise_native
    rer = 1 / np.sqrt(2 * np.pi * (c4 * res + blur ** 2))  # scale the native blur by res since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe5_deriv_6(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = noise + noise_native
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe5_deriv_6_nq(params, distortion_vector):
    # update from giqe5_deriv_6 to add noise in quadrature
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe5_deriv_7(params, distortion_vector):

    """
    same as v6, except that noise-rer cross term is removed (basically a sensitivity analysis).

    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = noise + noise_native
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same

    return y


def giqe5_deriv_7_nq(params, distortion_vector):

    """
    same as v6, except that noise-rer cross term is removed (basically a sensitivity analysis)

    update from giqe5_deriv_7 to add noise in quadrature
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same

    return y


def giqe5_deriv_8(params, distortion_vector):

    """
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time). Still no
    cross term.
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer)**4 + c6 * noise  # trying to keep the coefficients named the same

    return y


def giqe5_deriv_9(params, distortion_vector):

    """
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time). Still no
    cross term.

    Includes RER-SNR cross term.
    """

    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer)**4 + c6 * noise

    return y


def power_law(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * res ** c2 + c3 * blur ** c4 + c5 * noise ** c6

    return y


def rer_0(params, distortion_vector):
    """
    If native blur is constant, c0 and c1 are redundant, but this function is designed to be extensible to
    true 2d distortion vectors in which native blur may vary
    """
    c0, c1, c2, c3 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = c0 - c1 * native_blur + c2 * np.exp(c3 * blur)

    return y


def rer_1(params, distortion_vector):
    """
    If native blur is constant, c0 and c1 are redundant, but this function is designed to be extensible to
    true 2d distortion vectors in which native blur may vary
    """
    c0, c1 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = c0 * np.exp(c1 * native_blur) * np.exp(c1 * blur)

    return y


def rer_2(params, distortion_vector):
    """
    This function is NOT extensible to a true 2d fit that incorporates native and secondary blur. Instead, c1 depends
    the extent of native blur in each fit.
    """
    c0, c1 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = c0 / (1 + c1 * blur)

    return y


def rer_3(params, distortion_vector):
    """
    This function is NOT extensible to a true 2d fit that incorporates native and secondary blur. Instead, c0 is an
    approximation of the native blur in each chip.
    """
    c0 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = 1 / (np.sqrt(2 * np.pi) * (c0 + blur))

    return y


def rer_4(params, distortion_vector):
    """
    This function is NOT extensible to a true 2d fit that incorporates native and secondary blur. Instead, c0 is an
    approximation of the native blur in each chip.
    """
    c0 = params
    native_blur, blur = distortion_vector[:, 0], distortion_vector[:, 1]
    y = 1 / (np.sqrt(2 * np.pi * (c0 + blur ** 2)))

    return y


def fit(x, y, distortion_ids=('res', 'blur', 'noise'), fit_key='giqe5_deriv'):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise Exception('distortion_ids must == (res, blur, noise)')

    fit_function, initial_params = _fit_functions[fit_key]
    w = Fitter(x, y, fit_function, initial_params).fit()

    return w


def apply_fit(w, x, fit_key):
    fit_function = _fit_functions[fit_key][0]
    return fit_function(w, x)


# giqe5_2 initials
_c0 = 0.5
_c1 = 0.3
_c2 = 0.2
_c3 = -0.1
_c4 = 2
_c5 = 5
_c6 = 0.5
_c7 = -0.01


_fit_functions = {
    'giqe5_deriv_9': (giqe5_deriv_9, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_deriv_8': (giqe5_deriv_8, (_c0, _c1, 1, 0.5, -0.01)),

    'giqe5_deriv_7_nq': (giqe5_deriv_7_nq, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe5_deriv_6_nq': (giqe5_deriv_6_nq, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),

    'giqe5_deriv_7': (giqe5_deriv_7, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe5_deriv_6': (giqe5_deriv_6, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_deriv_5': (giqe5_deriv_5, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_deriv_4': (giqe5_deriv_4, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    # 'giqe5_deriv_3': giqe5_deriv_3,
    'giqe5_deriv_2': (giqe5_deriv_2,  (_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7)),
    'giqe5_deriv': (giqe5_deriv, (0.5, 0.3, 0.3, -1, 0.3, -0.14)),
    'power_law': (power_law, (0.5, 0.5, 1, -0.1, 1, -0.05, 1)),

    'rer_0': (rer_0,  (0.9, 0.25, 1, -1)),
    'rer_1': (rer_1, (0.9, -1)),
    'rer_2': (rer_2, (1, 1)),
    'rer_3': (rer_3, (1, )),
    'rer_4': (rer_4, (1, ))
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
