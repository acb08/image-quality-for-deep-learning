"""
Function for fits to be used when there is not a linear transform to enable fitting by singular value decomposition
"""
import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
from scipy.special import erf


def discrete_sampling_rer_model(sigma_blur):
    return erf(1 / (2 * np.sqrt(2) * sigma_blur))


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
    y = giqe3_deriv(params, x)
    return y, y + np.random.randn(len(y)) * sigma


def giqe3_deriv(params, distortion_vector):
    """
    Previously known as giqe5_deriv()
    """

    c0, c1, c2, c3, c4, c5 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(blur) + c4 * np.log10(blur) \
        + c5 * noise

    return y


def giqe3_deriv_2(params, distortion_vector):
    """
    Previously known as giqe5_deriv_2()
    """
    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    rer = c4 / (1 + c5 * blur)
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c6 * np.log10(rer) \
        + c7 * noise

    return y


def giqe3_deriv_3(params, distortion_vector):
    """
    Previously known as giqe5_deriv_3()
    """
    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    rer = c4 / (1 + c5 * blur / res)
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c6 * np.log10(rer) \
        + c7 * noise

    return y


def giqe3_deriv_4(params, distortion_vector):
    """
    Previously known as giqe5_deriv_4()
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    rer = 1 / np.sqrt(2 * np.pi * (c4 * res + blur ** 2))  # scale the native blur by res since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe3_deriv_5(params, distortion_vector):
    """
    Previously known as giqe5_deriv_5()
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = noise + noise_native
    rer = 1 / np.sqrt(2 * np.pi * (c4 * res + blur ** 2))  # scale the native blur by res since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe3_deriv_6(params, distortion_vector):
    """
    Previously known as giqe5_deriv_6()
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = noise + noise_native
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe3_deriv_6_nq(params, distortion_vector):
    """
    Previously known as giqe5_deriv_6_nq()
    """
    # update from giqe3_deriv_6 to add noise in quadrature
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
        + c6 * noise

    return y


def giqe3_deriv_7(params, distortion_vector):
    """
    same as v6, except that noise-rer cross term is removed (basically a sensitivity analysis).

    Previously known as giqe5_deriv_7()
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = noise + noise_native
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same

    return y


def giqe3_deriv_7_nq(params, distortion_vector):

    """
    same as v6, except that noise-rer cross term is removed (basically a sensitivity analysis)

    update from giqe3_deriv_7 to add noise in quadrature

    Previously known as giqe5_deriv_7_nq()
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
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time).

    Includes RER-SNR cross term.
    """

    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer)**4 + c6 * noise

    return y


def giqe5_deriv_10(params, distortion_vector):

    """
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time).

    No cross term.

    Uses discrete sampling (error function) RER model.
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    sigma_combined = np.sqrt((c4 * res)**2 + blur**2)
    rer = discrete_sampling_rer_model(sigma_combined)
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer)**4 + c6 * noise  # trying to keep the coefficients named the same

    return y


def giqe5_deriv_11(params, distortion_vector):

    """
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time).

    Includes RER-SNR cross term.

    Uses discrete sampling (error function) RER model.
    """

    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    sigma_combined = np.sqrt((c4 * res)**2 + blur**2)
    rer = discrete_sampling_rer_model(sigma_combined)
    # rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer)**4 + c6 * noise

    return y


def giqe3_deriv_12(params, distortion_vector):

    """
    Same as v7_nq, with RER updated for discrete sampling.

    No RER-SNR cross term

    Previously known as giqe5_deriv_12()
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    sigma_combined = np.sqrt((c4 * res) ** 2 + blur ** 2)
    rer = discrete_sampling_rer_model(sigma_combined)
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same

    return y


def power_law(params, distortion_vector):
    """
    No distortion mapping, direct application in fit.
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * res ** c2 + c3 * blur ** c4 + c5 * noise ** c6

    return y


def power_law_2(params, distortion_vector):
    """
    Distortions mapped to GIQE variables, with noise added in quadrature and blur mapped to RER va discrete sampling
    erf
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    sigma_combined = np.sqrt((2 * res)**2 + blur**2)
    rer = discrete_sampling_rer_model(sigma_combined)
    noise = np.sqrt(1 + noise**2)

    y = c0 + c1 * res ** c2 + c3 * rer ** c4 + c5 * noise ** c6

    return y


def power_law_3(params, distortion_vector):
    """
    Noise added in quadrature, no mapping to RER
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise = np.sqrt(1 + noise ** 2)
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


def fit(x, y, distortion_ids=('res', 'blur', 'noise'), fit_key='giqe3_deriv'):

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
    'giqe3_deriv_12': (giqe3_deriv_12, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe5_deriv_11': (giqe5_deriv_11, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_deriv_10': (giqe5_deriv_10, (_c0, _c1, 1, 0.5, -0.01)),

    'giqe5_deriv_9': (giqe5_deriv_9, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_deriv_8': (giqe5_deriv_8, (_c0, _c1, 1, 0.5, -0.01)),

    'giqe3_deriv_7_nq': (giqe3_deriv_7_nq, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_deriv_6_nq': (giqe3_deriv_6_nq, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),

    'giqe3_deriv_7': (giqe3_deriv_7, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_deriv_6': (giqe3_deriv_6, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe3_deriv_5': (giqe3_deriv_5, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe3_deriv_4': (giqe3_deriv_4, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    # 'giqe3_deriv_3': giqe3_deriv_3,
    'giqe3_deriv_2': (giqe3_deriv_2, (_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7)),
    'giqe3_deriv': (giqe3_deriv, (0.5, 0.3, 0.3, -1, 0.3, -0.14)),
    'power_law': (power_law, (0.5, 0.5, 0.5, -0.1, 0.5, -0.05, 0.5)),
    'power_law_2': (power_law_2, (0.5, 0.5, 0.5, -0.1, 0.5, -0.05, 0.5)),
    'power_law_3': (power_law_3, (0.5, 0.5, 0.5, -0.1, 0.5, -0.05, 0.5)),

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

    _y_fit = giqe3_deriv(_p_fit, _x0)

    plt.figure()
    plt.scatter(np.arange(len(_data)), _data)
    plt.plot(_y_fit)
    plt.plot(_truth)
    plt.show()

    print(_p_fit)
