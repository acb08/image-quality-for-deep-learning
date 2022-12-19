import numpy as np
from scipy.special import erf
from src.d00_utils.definitions import PROJECT_ID
from src.d04_analysis.fit_functions_rer import rer_0, rer_1, rer_2, rer_3, rer_4


LORENTZ_TERMS = (0.2630388847587775, -0.4590111280646474)  # correction for Gaussian to account for pixel xfer function
NATIVE_NOISE_ESTIMATE = 1


def _n1(n):
    return NATIVE_NOISE_ESTIMATE + n


def _n2(n):
    return np.sqrt(NATIVE_NOISE_ESTIMATE ** 2 + n ** 2)


# def _rer1(blur, res, c):
#     """
#     Obsolete.
#     """
#     combined_blur = _combined_blur_res_linear(blur, res, c)
#     return 1 / (np.sqrt(2 * np.pi) * combined_blur)


def _rer2(blur, res, c):
    combined_blur = _combined_blur_res_squared(blur, res, c)
    return 1 / (np.sqrt(2 * np.pi) * combined_blur)


def _rer3(blur, res, c,):
    combined_blur = _combined_blur_res_squared(blur, res, c)
    return discrete_sampling_rer_model(combined_blur, apply_blur_correction=False)


def _rer4(blur, res, c,):
    combined_blur = _combined_blur_res_squared(blur, res, c)
    return discrete_sampling_rer_model(combined_blur, apply_blur_correction=True)


def _combined_blur_res_linear(blur, res, c):
    raise DeprecationWarning('Obsolete function replaced by _combined_blur_res_squared()')
    # return np.sqrt(c * res + blur**2)


def _combined_blur_res_squared(blur, res, c):
    return np.sqrt((c * res)**2 + blur**2)


def _giqe5(res, rer, noise, params):
    c0, c1, c2, c3, c4, c5, c6 = params
    return (c0 + c1 * np.log10(1 / res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) ** 4 +
            c6 * noise)


def _giqe3(res, rer, noise, params):
    c0, c1, c4, c5, c6 = params  # trying to match coefficient names between giqe3 and giqe5
    return c0 + c1 * np.log10(1 / res) + c5 * np.log10(rer) + c6 * noise


def discrete_sampling_rer_model(sigma_blur, apply_blur_correction=False):
    if apply_blur_correction:
        corrected_blur = apply_xfer_function_correction(sigma_blur)
        return erf(1 / (2 * np.sqrt(2) * corrected_blur))
    else:
        return erf(1 / (2 * np.sqrt(2) * sigma_blur))


def apply_xfer_function_correction(sigma):
    """
    Applies a correction to Gaussian blur standard deviation to incorporate the effects of the pixel transfer function,
    which has a substantial impact at low blur values.
    """
    b = LORENTZ_TERMS[0]
    m = LORENTZ_TERMS[1]

    correction = b / (np.pi * (sigma - m)**2 + b**2)
    return sigma + correction


def exp_b4n0(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b4n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n1(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b4n2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n2(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b3n0(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b3n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n1(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b3n2(params, distortion_vector):
    """
    Distortions mapped to GIQE variables, with noise added in quadrature and blur mapped to RER va discrete sampling
    erf
    """
    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n2(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b2n0(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b2n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n1(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


def exp_b2n2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n2(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)

    return y


# def exp_b1n0(params, distortion_vector):
#     """
#     Obsolete.
#     """
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#
#     rer = _rer1(blur, res, c4)
#
#     y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)
#
#     return y


# def exp_b1n1(params, distortion_vector):
#     """
#     Obsolete.
#     """
#
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#
#     rer = _rer1(blur, res, c4)
#     noise = _n1(noise)
#
#     y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)
#
#     return y


# def exp_b1n2(params, distortion_vector):
#     """
#     Obsolete.
#     """
#
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#
#     rer = _rer1(blur, res, c4)
#     noise = _n2(noise)
#
#     y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c5 * rer) + c6 * np.exp(c7 * noise)
#
#     return y


def exp_b0n0(params, distortion_vector):
    """
    Distortions mapped to GIQE variables, with noise added in quadrature and blur mapped to RER va discrete sampling
    erf
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c4 * blur) + c5 * np.exp(c6 * noise)

    return y


def exp_b0n1(params, distortion_vector):
    """
    Distortions mapped to GIQE variables, with noise added in quadrature and blur mapped to RER va discrete sampling
    erf
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n1(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c4 * blur) + c5 * np.exp(c6 * noise)

    return y


def exp_b0n2(params, distortion_vector):
    """
    Distortions mapped to GIQE variables, with noise added in quadrature and blur mapped to RER va discrete sampling
    erf
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n2(noise)

    y = c0 + c1 * np.exp(c2 * res) + c3 * np.exp(c4 * blur) + c5 * np.exp(c6 * noise)

    return y


def pl_b0n0(params, distortion_vector):
    """
    No distortion mapping, direct application in fit.
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * res ** c2 + c3 * blur ** c4 + c5 * noise ** c6

    return y


def pl_b0n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n1(noise)

    y = c0 + c1 * res ** c2 + c3 * blur ** c4 + c5 * noise ** c6

    return y


def pl_b0n2(params, distortion_vector):
    """
    Noise added in quadrature, no mapping to RER
    """
    c0, c1, c2, c3, c4, c5, c6 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n2(noise)

    y = c0 + c1 * res ** c2 + c3 * blur ** c4 + c5 * noise ** c6

    return y


# def pl_b1n0(params, distortion_vector):
#     """
#     Obsolete.
#     """
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#
#     rer = _rer1(blur, noise, c4)
#
#     y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7
#
#     return y


# def pl_b1n1(params, distortion_vector):
#     """
#     Obsolete.
#     """
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#
#     rer = _rer1(blur, noise, c4)
#     noise = _n1(noise)
#
#     y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7
#
#     return y


# def pl_b1n2(params, distortion_vector):
#     """
#     Obsolete.
#     """
#
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#
#     rer = _rer1(blur, noise, c4)
#     noise = _n2(noise)
#
#     y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7
#
#     return y


def pl_b2n0(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    __ = c0
    __ = c1 * res ** c2
    __ = c3 * rer ** c5
    __ = c6 * noise ** c7

    return y


def pl_b2n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n1(noise)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def pl_b2n2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n2(noise)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def pl_b3n0(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    __ = c0
    __ = c1 * res ** c2
    __ = c3 * rer ** c5
    __ = c6 * noise ** c7

    return y


def pl_b3n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n1(noise)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def pl_b3n2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n2(noise)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def pl_b4n0(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def pl_b4n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n1(noise)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def pl_b4n2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6, c7 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n2(noise)

    y = c0 + c1 * res ** c2 + c3 * rer ** c5 + c6 * noise ** c7

    return y


def giqe35_b0n0(params, distortion_vector):
    """
    Previously known as giqe5_deriv()
    """

    c0, c1, c2, c3, c4, c5 = params
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(blur) + c4 * np.log10(blur) \
        + c5 * noise

    return y


# def giqe3_deriv_2(params, distortion_vector):
#     """
#     Previously known as giqe5_deriv_2()
#     """
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     rer = c4 / (1 + c5 * blur)
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp_b3n2(c3 * noise)) * np.log10(rer) + c6 * np.log10(rer) \
#         + c7 * noise
#
#     return y


# def giqe3_deriv_3(params, distortion_vector):
#     """
#     Previously known as giqe5_deriv_3()
#     """
#     c0, c1, c2, c3, c4, c5, c6, c7 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     rer = c4 / (1 + c5 * blur / res)
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp_b3n2(c3 * noise)) * np.log10(rer) + c6 * np.log10(rer) \
#         + c7 * noise
#
#     return y


# def giqe3_n0b1(params, distortion_vector):
#     """
#     Previously known as giqe5_deriv_4()
#     """
#     c0, c1, c2, c3, c4, c5, c6 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     # rer = 1 / np.sqrt(2 * np.pi * (c4 * res + blur ** 2))  # scale native blur by res since down-sampling sharpens
#     rer = _rer1(blur, res, c4)
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp_b3n2(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
#         + c6 * noise
#
#     return y


# def giqe3_deriv_5(params, distortion_vector):
#     """
#     Previously known as giqe5_deriv_5()
#     """
#     c0, c1, c2, c3, c4, c5, c6 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     noise = noise + noise_native
#     rer = 1 / np.sqrt(2 * np.pi * (c4 * res + blur ** 2))  # scale the native blur by res since down-sampling sharpens
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp_b3n2(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
#         + c6 * noise
#
#     return y


# def giqe3_deriv_6(params, distortion_vector):
#     """
#     Previously known as giqe5_deriv_6()
#     """
#     c0, c1, c2, c3, c4, c5, c6 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     noise = noise + noise_native
#     rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp_b3n2(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
#         + c6 * noise
#
#     return y


# def giqe3_deriv_6_nq(params, distortion_vector):
#     """
#     Previously known as giqe5_deriv_6_nq()
#     """
#     # update from giqe3_deriv_6 to add noise in quadrature
#     c0, c1, c2, c3, c4, c5, c6 = params
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     noise = np.sqrt(noise ** 2 + noise_native ** 2)
#     rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp_b3n2(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer) \
#         + c6 * noise
#
#     return y

def giqe3_b0n0(params, distortion_vector):

    c0, c1, c5, c6 = params  # keeping parameter names consistent with v6
    c4 = None  # by convention, I have used c4 to represent native blur in RER functions (not used here)
    params = (c0, c1, c4, c5, c6)
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    return _giqe3(res, blur, noise, params)


def giqe3_b0n1(params, distortion_vector):

    c0, c1, c5, c6 = params  # keeping parameter names consistent with v6
    c4 = None  # by convention, I have used c4 to represent native blur in RER functions (not used here)
    params = (c0, c1, c4, c5, c6)
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n1(noise)

    return _giqe3(res, blur, noise, params)


def giqe3_b0n2(params, distortion_vector):

    c0, c1, c5, c6 = params  # keeping parameter names consistent with v6
    c4 = None  # by convention, I have used c4 to represent native blur in RER functions (not used here)
    params = (c0, c1, c4, c5, c6)
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n2(noise)

    return _giqe3(res, blur, noise, params)


def giqe3_b2n0(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)

    return _giqe3(res, rer, noise, params)


def giqe3_b2n1(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n1(noise)

    return _giqe3(res, rer, noise, params)

    # y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same
    #
    # return y


def giqe3_b2n2(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n2(noise)

    return _giqe3(res, rer, noise, params)
    #
    # y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same
    #
    # return y


def giqe3_b3n0(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)

    return _giqe3(res, rer, noise, params)


def giqe3_b3n1(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n1(noise)

    return _giqe3(res, rer, noise, params)


def giqe3_b3n2(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n2(noise)

    return _giqe3(res, rer, noise, params)


def giqe3_b4n0(params, distortion_vector):

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)

    return _giqe3(res, rer, noise, params)


def giqe3_b4n1(params, distortion_vector):
    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n1(noise)

    return _giqe3(res, rer, noise, params)


def giqe3_b4n2(params, distortion_vector):
    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n2(noise)

    return _giqe3(res, rer, noise, params)


def giqe5_b2b2_nct(params, distortion_vector):

    """
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time). Still no
    cross term.
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    # noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    # noise = np.sqrt(noise ** 2 + noise_native ** 2)
    # noise = _n2(noise)

    noise = _n2(noise)
    rer = _rer2(blur, res, c4)

    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer)**4 + c6 * noise  # trying to keep the coefficients named the same

    return y


# def giqe5_b2n2(params, distortion_vector):
#
#     """
#     Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time).
#
#     Includes RER-SNR cross term.
#     """
#
#     c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     # noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     # noise = np.sqrt(noise ** 2 + noise_native ** 2)
#     noise = _n2(noise)
#     # rer = 1 / np.sqrt(2 * np.pi * ((c4 * res) ** 2 + blur ** 2))  # scale by res sq since down-sampling sharpens
#     rer = _rer2(blur, res, c4)
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer)**4 + c6 * noise
#
#     return y


def giqe5_b3n2_nct(params, distortion_vector):

    """
    Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time).

    No cross term.

    Uses discrete sampling (error function) RER model.
    """

    c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
    noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
    noise = np.sqrt(noise ** 2 + noise_native ** 2)
    # sigma_combined = np.sqrt((c4 * res)**2 + blur**2)
    # rer = discrete_sampling_rer_model(sigma_combined, apply_blur_correction=False)
    rer = _rer3(blur, res, c4)
    y = c0 + c1 * np.log10(res) + c5 * np.log10(rer)**4 + c6 * noise  # trying to keep the coefficients named the same

    return y


# def giqe5_b3n2(params, distortion_vector):
#
#     """
#     Incorporates raising the RER term to the fourth power (which I missed for an embarrassingly long time).
#
#     Includes RER-SNR cross term.
#
#     Uses discrete sampling (error function) RER model.
#     """
#
#     c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     # noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     # noise = np.sqrt(noise ** 2 + noise_native ** 2)
#     noise = _n2(noise)
#     # sigma_combined = np.sqrt((c4 * res)**2 + blur**2)
#     # rer = discrete_sampling_rer_model(sigma_combined, apply_blur_correction=False)
#     rer = _rer3(blur, res, c4)
#     y = c0 + c1 * np.log10(res) + c2 * (1 - np.exp(c3 * noise)) * np.log10(rer) + c5 * np.log10(rer)**4 + c6 * noise
#
#     return y


# def giqe3_b3n2(params, distortion_vector):
#
#     """
#     Same as v7_nq, with RER updated for discrete sampling.
#
#     No RER-SNR cross term
#
#     Previously known as giqe5_deriv_12()
#     """
#
#     c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     # noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     # noise = np.sqrt(noise ** 2 + noise_native ** 2)
#     noise = _n2(noise)
#     # sigma_combined = np.sqrt((c4 * res) ** 2 + blur ** 2)
#     # rer = discrete_sampling_rer_model(sigma_combined, apply_blur_correction=False)
#     rer = _rer3(blur, res, c4)
#     y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same
#
#     return y


# def giqe3_b4n2(params, distortion_vector):
#
#     """
#     Derived from as v7_nq and v12, with RER updated for discrete sampling AND blur correction applied to account for
#     pixel transfer function.
#
#     No RER-SNR cross term
#     """
#
#     c0, c1, c4, c5, c6 = params  # keeping parameter names consistent with v6
#     res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]
#     # noise_native = 1  # counts, estimated/sanity checked using very simple model in src.analysis.noise_estimate
#     # noise = np.sqrt(noise ** 2 + noise_native ** 2)
#     noise = _n2(noise)
#     # sigma_combined = np.sqrt((c4 * res) ** 2 + blur ** 2)
#     # rer = discrete_sampling_rer_model(sigma_combined, apply_blur_correction=True)
#     rer = _rer4(blur, res, c4)
#     y = c0 + c1 * np.log10(res) + c5 * np.log10(rer) + c6 * noise  # trying to keep the coefficients named the same
#
#     return y


def giqe5_b0n0(params, distortion_vector):
    c0, c1, c2, c3, c5, c6 = params  # keeping parameter names consistent with v6
    c4 = None
    params = (c0, c1, c2, c3, c4, c5, c6)
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    return _giqe5(res, blur, noise, params)


def giqe5_b0n1(params, distortion_vector):
    c0, c1, c2, c3, c5, c6 = params  # keeping parameter names consistent with v6
    c4 = None
    params = (c0, c1, c2, c3, c4, c5, c6)
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n1(noise)

    return _giqe5(res, blur, noise, params)


def giqe5_b0n2(params, distortion_vector):
    c0, c1, c2, c3, c5, c6 = params  # keeping parameter names consistent with v6
    c4 = None
    params = (c0, c1, c2, c3, c4, c5, c6)
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    noise = _n2(noise)

    return _giqe5(res, blur, noise, params)


def giqe5_b2n0(params, distortion_vector):
    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)

    return _giqe5(res, rer, noise, params)


def giqe5_b2n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n1(noise)

    return _giqe5(res, rer, noise, params)


def giqe5_b2n2(params, distortion_vector):
    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer2(blur, res, c4)
    noise = _n2(noise)

    return _giqe5(res, rer, noise, params)


def giqe5_b3n0(params, distortion_vector):
    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)

    return _giqe5(res, rer, noise, params)


def giqe5_b3n1(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n1(noise)

    return _giqe5(res, rer, noise, params)


def giqe5_b3n2(params, distortion_vector):
    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer3(blur, res, c4)
    noise = _n2(noise)

    return _giqe5(res, rer, noise, params)


def giqe5_b4n0(params, distortion_vector):
    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)

    return _giqe5(res, rer, noise, params)


def giqe5_b4n1(params, distortion_vector):
    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n1(noise)

    return _giqe5(res, rer, noise, params)


def giqe5_b4n2(params, distortion_vector):

    c0, c1, c2, c3, c4, c5, c6 = params  # keeping parameter names consistent with v6
    res, blur, noise = distortion_vector[:, 0], distortion_vector[:, 1], distortion_vector[:, 2]

    rer = _rer4(blur, res, c4)
    noise = _n2(noise)

    return _giqe5(res, rer, noise, params)


def fit_key(functional_form, blur_map, noise_map):
    return f'{functional_form}_{blur_map}{noise_map}'


def generate_fit_keys(functional_forms, blur_maps, noise_maps):

    fit_keys = []
    for functional_form in functional_forms:
        for blur_map in blur_maps:
            for noise_map in noise_maps:
                key = fit_key(functional_form, blur_map, noise_map)
                fit_keys.append(key)

    return fit_keys


# giqe5_2 initials
_c0 = 0.5
_c1 = - 0.3
_c2 = 0.2
_c3 = -0.1
_c4 = 2
_c5 = 5
_c6 = 0.5
_c7 = -0.01

_fit_funcs_sat6_start_params = {
    'exp_b3n0': (exp_b3n0, (0.7, -0.4, -1.8, -0.2, 1, -3, 0.3, -0.01)),
    'exp_b2n2': (exp_b2n2, (0.7, -0.4, -1.8, -0.2, 1, -3, 0.3, -0.01)),
    'exp_b4n1': (exp_b4n1, (0.7, -0.4, -1.8, -0.2, 1, -3, 0.3, -0.01)),
}  # not some fits that convert for Places365 do not converge for SAT6, so using SAT6-specific starting parameters
# where needed.

fit_functions = {
    'exp_b4n2': (exp_b4n2, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),
    'exp_b4n1': (exp_b4n1, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),
    'exp_b4n0': (exp_b4n0, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),

    'exp_b3n2': (exp_b3n2, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),
    'exp_b3n1': (exp_b3n1, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),
    'exp_b3n0': (exp_b3n0, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),

    'exp_b2n2': (exp_b2n2, (_c0, -0.1, -1, -0.1, 1, -2, 0.1, -0.01)),
    'exp_b2n1': (exp_b2n1, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),
    'exp_b2n0': (exp_b2n0, (_c0, -0.1, -1, -0.1, 1, -1, 0.1, -0.01)),

    'exp_b0n2': (exp_b0n2, (_c0, -0.1, -1, -0.1, -0.5, 0.1, -0.01)),
    'exp_b0n1': (exp_b0n1, (_c0, -0.1, -1, -0.1, -0.5, 0.1, -0.01)),
    'exp_b0n0': (exp_b0n0, (_c0, -0.1, -1, -0.1, -0.5, 0.1, -0.01)),

    'pl_b0n0': (pl_b0n0, (0.5, 0.5, 0.5, -0.1, 0.5, -0.05, 0.5)),
    'pl_b0n1': (pl_b0n1, (0.5, 0.5, 0.5, -0.1, 0.5, -0.05, 0.5)),
    'pl_b0n2': (pl_b0n2, (0.5, 0.5, 0.5, -0.1, 0.5, -0.05, 0.5)),

    'pl_b2n0': (pl_b2n0, (0.5, 0.5, 0.5, 1, 2, 0.5, -0.05, 0.5)),  # (0.5, 0.5, 0.5, 0.1, 1, 1, -0.05, 0.5)),
    'pl_b2n1': (pl_b2n1, (0.5, 0.5, 0.5, 1, 2, 0.5, -0.05, 0.5)),
    'pl_b2n2': (pl_b2n2, (0.5, 0.5, 0.5, 1, 2, 0.5, -0.05, 0.5)),

    'pl_b3n0': (pl_b3n0, (0.5, 0.5, 0.5, 1, 2, 0.5, -2, 1)),
    'pl_b3n1': (pl_b3n1, (0.5, 0.5, 0.5, 1, 2, 0.5, -2, 1)),
    'pl_b3n2': (pl_b3n2, (0.5, 0.5, 0.5, 1, 2, 0.5, -2, 1)),

    'pl_b4n0': (pl_b4n0, (0.5, 0.5, 0.5, 1, 2, 0.5, -2, 1)),
    'pl_b4n1': (pl_b4n1, (0.5, 0.5, 0.5, 1, 2, 0.5, -2, 1)),
    'pl_b4n2': (pl_b4n2, (0.5, 0.5, 0.5, 1, 2, 0.5, -2, 1)),  # (0.5, 0.5, 0.5, -0.1, 0.5, 2, -0.05, 0.5)

    'giqe3_b0n0': (giqe3_b0n0, (_c0, _c1, 0.5, -0.01)),
    'giqe3_b0n1': (giqe3_b0n1, (_c0, _c1, -0.5, -0.01)),
    'giqe3_b0n2': (giqe3_b0n2, (_c0, _c1, -0.5, -0.01)),

    'giqe3_b2n0': (giqe3_b2n0, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_b2n1': (giqe3_b2n1, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_b2n2': (giqe3_b2n2, (_c0, _c1, 1, 0.5, -0.01)),

    'giqe3_b3n0': (giqe3_b3n0, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_b3n1': (giqe3_b3n1, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_b3n2': (giqe3_b3n2, (_c0, _c1, 1, 0.5, -0.01)),

    'giqe3_b4n0': (giqe3_b4n0, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_b4n1': (giqe3_b4n1, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe3_b4n2': (giqe3_b4n2, (_c0, _c1, 1, 0.5, -0.01)),

    # 'giqe5_b0n0': (giqe5_b0n0, (-0.5, 0.3, 0.1, -0.1, -0.1, -0.01)),  # before the log(1/r) update
    # 'giqe5_b0n1': (giqe5_b0n1, (-0.5, 0.3, 0.1, -0.1, -0.1, -0.01)),
    # 'giqe5_b0n2': (giqe5_b0n2, (-0.5, 0.3, 0.1, -0.1, -0.1, -0.01)),

    'giqe5_b0n0': (giqe5_b0n0, (_c0, _c1, 0.1, -0.1, -0.1, -0.01)),
    'giqe5_b0n1': (giqe5_b0n1, (_c0, _c1, 0.1, -0.1, -0.1, -0.01)),
    'giqe5_b0n2': (giqe5_b0n2, (_c0, _c1, 0.1, -0.1, -0.1, -0.01)),

    # 'giqe5_b2n0': (giqe5_b2n0, (-0.5, 0.3, 0.1, -0.1, 2, -0.1, -0.01)),  # before the log(1/r) update
    # 'giqe5_b2n1': (giqe5_b2n1, (-0.5, 0.3, 0.1, -0.1, 2, -0.1, -0.01)),
    # 'giqe5_b2n2': (giqe5_b2n2, (-0.5, 0.3, 0.1, -0.1, 2, -0.1, -0.01)),

    'giqe5_b2n0': (giqe5_b2n0, (_c0, _c1, 0.1, -0.1, 2, -0.1, -0.01)),
    'giqe5_b2n1': (giqe5_b2n1, (_c0, _c1, 0.1, -0.1, 2, -0.1, -0.01)),
    'giqe5_b2n2': (giqe5_b2n2, (_c0, _c1, 0.1, -0.1, 2, -0.1, -0.01)),

    # 'giqe5_b3n0': (giqe5_b3n2, (-0.5, 0.3, 0.1, -0.1, 2, -0.1, -0.01)), # before the log(1/r) update
    # 'giqe5_b3n1': (giqe5_b3n1, (-0.5, 0.3, 0.1, -0.1, 2, -0.1, -0.01)),
    # 'giqe5_b3n2': (giqe5_b3n2, (-0.5, 0.3, 0.1, -0.1, 2, -0.1, -0.01)),

    'giqe5_b3n0': (giqe5_b3n2, (_c0, _c1, 0.1, -0.1, 2, -0.1, -0.01)),
    'giqe5_b3n1': (giqe5_b3n1, (_c0, _c1, 0.1, -0.1, 2, -0.1, -0.01)),
    'giqe5_b3n2': (giqe5_b3n2, (_c0, _c1, 0.1, -0.1, 2, -0.1, -0.01)),

    'giqe5_b4n0': (giqe5_b4n0, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_b4n1': (giqe5_b4n1, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    'giqe5_b4n2': (giqe5_b4n2, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),


    'giqe5_b3n2_nct': (giqe5_b3n2_nct, (_c0, _c1, 1, 0.5, -0.01)),
    'giqe5_b2b2_nct': (giqe5_b2b2_nct, (_c0, _c1, 1, 0.5, -0.01)),


    # 'giqe3_deriv_6_nq': (giqe3_deriv_6_nq, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),


    # 'giqe3_deriv_6': (giqe3_deriv_6, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    # 'giqe3_deriv_5': (giqe3_deriv_5, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    # 'giqe3_n0b1': (giqe3_n0b1, (_c0, _c1, _c2, _c3, 1, 0.5, -0.01)),
    # 'giqe3_deriv_3': giqe3_deriv_3,
    # 'giqe3_deriv_2': (giqe3_deriv_2, (_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7)),
    # 'giqe35_b0n0': (giqe35_b0n0, (0.5, 0.3, 0.3, -1, 0.3, -0.14)),


    'rer_0': (rer_0, (0.9, 0.25, 1, -1)),
    'rer_1': (rer_1, (0.9, -1)),
    'rer_2': (rer_2, (1, 1)),
    'rer_3': (rer_3, (1,)),
    'rer_4': (rer_4, (1,))
}

if PROJECT_ID[:4] == 'sat6':
    fit_functions.update(_fit_funcs_sat6_start_params)


if __name__ == '__main__':

    _blur = np.random.rand(30) * 4.5 + 0.5
    _res = np.random.rand(30) * 0.75 + 0.25
    _c = np.random.rand(30) * 2 + 0.5

    _rer_2_test_func = _rer2(_blur, _res, _c)
    _rer_2_test_manual = 1 / np.sqrt(2 * np.pi * ((_c * _res) ** 2 + _blur ** 2))
    _diff_2 = _rer_2_test_func - _rer_2_test_manual
    print("2:", np.max(_diff_2), np.mean(_diff_2))

