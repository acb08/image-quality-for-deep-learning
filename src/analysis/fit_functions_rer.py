import numpy as np


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
