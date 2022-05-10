from src.d00_utils.classes import VariableResolution, VariablePoissonNoiseChannelReplicated
import numpy as np
from src.d00_utils.definitions import DISTORTION_RANGE, NATIVE_RESOLUTION
from torchvision import transforms


def rt_fr_s6():

    """
    Resolution transform for full range of distortion transforms (25-100%, or 7-28 pixels).

    Intended for use in a dataloader.
    """

    min_size = DISTORTION_RANGE['sat6']['res'][0]
    max_size = DISTORTION_RANGE['sat6']['res'][1]

    if max_size != NATIVE_RESOLUTION:
        raise Exception('max_size does not match native resolution as set in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes)


def bt_fr_s6():
    """
    variable blur transform covering full range of the blur spaces
    """
    kernel_size = DISTORTION_RANGE['sat6']['blur'][0]
    std_min = DISTORTION_RANGE['sat6']['blur'][1]
    std_max = DISTORTION_RANGE['sat6']['blur'][2]
    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def nt_fr_s6():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the sat6 nose range scaled
    by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson.

    Intended to be used in dataloader via Transforms.compose().
    """
    lambda_poisson_range = DISTORTION_RANGE['sat6']['noise']
    clamp = True
    return VariablePoissonNoiseChannelReplicated(lambda_poisson_range, clamp)


# places365 transforms

def rt_fr_pl():

    """
    Resolution transform for full range of distortion transforms (10-100%).

    Intended for use in a dataloader.
    """

    min_res = DISTORTION_RANGE['places365']['res'][0]
    min_size = int(min_res * NATIVE_RESOLUTION)
    max_res = DISTORTION_RANGE['places365']['res'][1]
    max_size = int(max_res * NATIVE_RESOLUTION)

    if max_size != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes)


def bt_fr_pl():
    """
    variable blur transform covering full range of the blur spaces
    """
    kernel_size = DISTORTION_RANGE['places365']['blur'][0]
    std_min = DISTORTION_RANGE['places365']['blur'][1]
    std_max = DISTORTION_RANGE['places365']['blur'][2]
    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def nt_fr_pl():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the sat6 nose range scaled
    by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    lambda_poisson_range = DISTORTION_RANGE['places365']['noise']
    clamp = True
    return VariablePoissonNoiseChannelReplicated(lambda_poisson_range, clamp)


tag_to_transform = {
    # sat6 transforms
    'rt_fr_s6': rt_fr_s6,

    'bt_fr_s6': bt_fr_s6,

    'nt_fr_s6': nt_fr_s6,

    # places transforms
    'rt_fr_pl': rt_fr_pl,

    'bt_fr_pl': bt_fr_pl,

    'nt_fr_pl': nt_fr_pl,
}
