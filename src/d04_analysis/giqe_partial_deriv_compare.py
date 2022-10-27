import numpy as np

GIQE_COEFFICIENTS = (
    9.57,  # A0
    -3.32,  # A1
    3.32,  # A2
    -1.9,  # A3
    -2,  # A4
    -1.8  # A5
)


def get_niirs(gsd, rer, snr):

    """

    :param gsd: ground sample distance (inches)
    :param rer:
    :param snr:
    :return:
    """

    a0 = GIQE_COEFFICIENTS[0]
    a1 = GIQE_COEFFICIENTS[1]
    a2 = GIQE_COEFFICIENTS[2]
    a3 = GIQE_COEFFICIENTS[3]
    a4 = GIQE_COEFFICIENTS[4]
    a5 = GIQE_COEFFICIENTS[5]
    pass
