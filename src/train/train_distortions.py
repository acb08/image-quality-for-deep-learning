from src.pre_processing.classes import VariableResolution, VariablePoissonNoiseChannelReplicated, \
    VariablePoissonNoiseIndependentChannel
import numpy as np
from src.utils.definitions import DISTORTION_RANGE, NATIVE_RESOLUTION, DISTORTION_RANGE_90
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

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_1_s6():

    """
    Resolution transform for lower half of full resolution range (i.e. the half of the resolution range with the greater
    degree of downsampling).

    Intended for use in a dataloader.
    """

    min_res_fr = DISTORTION_RANGE['sat6']['res'][0]
    min_size = min_res_fr
    max_res_fr = DISTORTION_RANGE['sat6']['res'][1]
    max_res = (min_res_fr + max_res_fr) / 2  # midpoint of resolution range
    max_size = int(max_res)

    if NATIVE_RESOLUTION != 28:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_0_s6():

    """
    Resolution transform for upper half of full resolution range (i.e. the half of the resolution range with the lesser
    degree of downsampling).

    Intended for use in a dataloader.
    """

    min_res_fr = DISTORTION_RANGE['sat6']['res'][0]
    max_res_fr = DISTORTION_RANGE['sat6']['res'][1]

    min_res = (min_res_fr + max_res_fr) / 2  # midpoint of resolution range
    min_size = int(min_res)
    max_size = max_res_fr

    if NATIVE_RESOLUTION != 28:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_ep_s6():
    """
    Resolution transform for the endpoint of the sat6 distortion space
    """

    min_res_fr = DISTORTION_RANGE['sat6']['res'][0]
    size = min_res_fr

    if NATIVE_RESOLUTION != 28:
        raise Exception('mismatch between max size and native resolution in project config')

    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)


def rt_mp_s6():
    """
    Resolution transform for the midpoint of the sat6 distortion space
    """

    min_res_fr = DISTORTION_RANGE['sat6']['res'][0]
    max_res_fr = DISTORTION_RANGE['sat6']['res'][1]
    size = int((min_res_fr + max_res_fr) / 2)  # midpoint of resolution range

    if NATIVE_RESOLUTION != 28:
        raise Exception('mismatch between max size and native resolution in project config')

    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)


def bt_fr_s6():
    """
    variable blur transform covering full range of the blur space
    """
    kernel_size = DISTORTION_RANGE['sat6']['blur'][0]
    std_min = DISTORTION_RANGE['sat6']['blur'][1]
    std_max = DISTORTION_RANGE['sat6']['blur'][2]
    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_0_s6():
    """
    variable blur transform covering lower half of sat6 full range blur space
    """
    kernel_size = DISTORTION_RANGE['sat6']['blur'][0]
    std_min_fr = DISTORTION_RANGE['sat6']['blur'][1]
    std_max_fr = DISTORTION_RANGE['sat6']['blur'][2]

    std_min = std_min_fr
    std_max = (std_min_fr + std_max_fr) / 2

    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_1_s6():
    """
    variable blur transform covering upper half of sat6 full range blur space
    """
    kernel_size = DISTORTION_RANGE['sat6']['blur'][0]
    std_min_fr = DISTORTION_RANGE['sat6']['blur'][1]
    std_max_fr = DISTORTION_RANGE['sat6']['blur'][2]

    std_min = (std_min_fr + std_max_fr) / 2
    std_max = std_max_fr

    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_ep_s6():
    """
    Blur transform for the endpoint of the sat6 distortion space
    """
    kernel_size = DISTORTION_RANGE['sat6']['blur'][0]
    std_max_fr = DISTORTION_RANGE['sat6']['blur'][2]
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std_max_fr)


def bt_mp_s6():
    """
    Blur transform for the midpoint of the sat6 distortion space
    """
    kernel_size = DISTORTION_RANGE['sat6']['blur'][0]
    std_min_fr = DISTORTION_RANGE['sat6']['blur'][1]
    std_max_fr = DISTORTION_RANGE['sat6']['blur'][2]
    std = (std_min_fr + std_max_fr) / 2
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_mp90_s6():
    """
    Blur transform for the midpoint of the sat6 distortion space
    """
    kernel_size = DISTORTION_RANGE_90['sat6']['blur'][0]
    std_min_fr = DISTORTION_RANGE_90['sat6']['blur'][1]
    std_max_fr = DISTORTION_RANGE_90['sat6']['blur'][2]
    std = (std_min_fr + std_max_fr) / 2
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def nt_fr_s6():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the sat6 noise range
    scaled by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_range = DISTORTION_RANGE['sat6']['noise']  #
    clamp = True
    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_0_s6():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the lower half of the
    sat6 noise range, scaled by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor =
    input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE['sat6']['noise']
    sigma_poisson_min = sigma_poisson_min_fr
    sigma_poisson_max = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    sigma_poisson_range = (sigma_poisson_min, sigma_poisson_max)
    clamp = True
    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_1_s6():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the upper half of the
    sat6 noise range, scaled by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor =
    input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE['sat6']['noise']
    sigma_poisson_min = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    sigma_poisson_max = sigma_poisson_max_fr
    sigma_poisson_range = (sigma_poisson_min, sigma_poisson_max)
    clamp = True
    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_ep_s6():
    """
    Noise transform at endpoint of sat6 distortion space
    """

    __, sigma_poisson_max_fr = DISTORTION_RANGE['sat6']['noise']
    clamp = True
    sigma_poisson_range = (sigma_poisson_max_fr, sigma_poisson_max_fr)

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_mp_s6():
    """
    Noise transform at midpoint of sat6 distortion space
    """

    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE['sat6']['noise']
    sigma_poisson_midpoint = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    clamp = True
    sigma_poisson_range = (sigma_poisson_midpoint, sigma_poisson_midpoint)

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)

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

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_fr_pl_cp():

    """
    Resolution transform for to parallel COCO full range training

    Intended for use in a dataloader.
    """

    min_res = DISTORTION_RANGE['coco']['res'][0]
    min_size = int(min_res * NATIVE_RESOLUTION)
    max_res = DISTORTION_RANGE['coco']['res'][1]
    max_size = int(max_res * NATIVE_RESOLUTION)

    if max_size != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_1_pl():

    """
    Resolution transform for lower half of full resolution range (i.e. the half of the resolution range with the greater
    degree of downsampling).

    Intended for use in a dataloader.
    """

    min_res_fr = DISTORTION_RANGE['places365']['res'][0]
    min_size = int(min_res_fr * NATIVE_RESOLUTION)
    max_res_fr = DISTORTION_RANGE['places365']['res'][1]
    max_res = (min_res_fr + max_res_fr) / 2  # midpoint of resolution range
    max_size = int(max_res * NATIVE_RESOLUTION)

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_0_pl():

    """
    Resolution transform for upper half of full resolution range (i.e. the half of the resolution range with the lesser
    degree of downsampling).

    Intended for use in a dataloader.
    """

    min_res_fr = DISTORTION_RANGE['places365']['res'][0]
    max_res_fr = DISTORTION_RANGE['places365']['res'][1]

    min_res = (min_res_fr + max_res_fr) / 2  # midpoint of resolution range
    min_size = int(min_res * NATIVE_RESOLUTION)
    max_size = int(max_res_fr * NATIVE_RESOLUTION)

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = list(np.arange(min_size, max_size + 1))

    return VariableResolution(sizes, interpolation_mode='bilinear', antialias=False)


def rt_ep_pl():
    """
    Resolution transform for hte endpoint of the sat6 distortion space
    """

    min_res_fr = DISTORTION_RANGE['places365']['res'][0]
    size = int(min_res_fr * NATIVE_RESOLUTION)

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)


def rt_ep90_pl():
    """
    Resolution transform for hte endpoint of the sat6 distortion space
    """

    min_res_fr = DISTORTION_RANGE_90['places365']['res'][0]
    size = int(min_res_fr * NATIVE_RESOLUTION)

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)


def rt_mp_pl():
    """
    Resolution transform for the midpoint of the places distortion space
    """
    min_res_fr = DISTORTION_RANGE['places365']['res'][0]
    max_res_fr = DISTORTION_RANGE['places365']['res'][1]

    res = (min_res_fr + max_res_fr) / 2  # midpoint of resolution range
    size = int(res * NATIVE_RESOLUTION)

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)


def rt_mp90_pl():
    """
    Resolution transform for the midpoint of the places distortion space
    """
    min_res_fr = DISTORTION_RANGE_90['places365']['res'][0]
    max_res_fr = DISTORTION_RANGE_90['places365']['res'][1]

    res = (min_res_fr + max_res_fr) / 2  # midpoint of resolution range
    size = int(res * NATIVE_RESOLUTION)

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    return transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)


def bt_fr_pl():
    """
    variable blur transform covering full range of the places blur space
    """
    kernel_size = DISTORTION_RANGE['places365']['blur'][0]
    std_min = DISTORTION_RANGE['places365']['blur'][1]
    std_max = DISTORTION_RANGE['places365']['blur'][2]
    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_fr_pl_cp():
    """
    Variable blur transform to parallel COCO full range training
    """
    kernel_size = 31
    std_min = DISTORTION_RANGE['coco']['blur'][0]
    std_max = DISTORTION_RANGE['coco']['blur'][1]
    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_0_pl():
    """
    variable blur transform covering the lower half of the place blur range
    """
    kernel_size = DISTORTION_RANGE['places365']['blur'][0]
    std_min_fr = DISTORTION_RANGE['places365']['blur'][1]
    std_max_fr = DISTORTION_RANGE['places365']['blur'][2]

    std_min = std_min_fr
    std_max = (std_min_fr + std_max_fr) / 2

    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_1_pl():
    """
    variable blur transform covering the upper half of the places blur range
    """
    kernel_size = DISTORTION_RANGE['places365']['blur'][0]
    std_min_fr = DISTORTION_RANGE['places365']['blur'][1]
    std_max_fr = DISTORTION_RANGE['places365']['blur'][2]

    std_min = (std_min_fr + std_max_fr) / 2
    std_max = std_max_fr

    std = (std_min, std_max)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_ep_pl():
    """
    Blur transform for the endpoint of the places365 distortion space
    """
    kernel_size = DISTORTION_RANGE['places365']['blur'][0]
    std_max_fr = DISTORTION_RANGE['places365']['blur'][2]
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std_max_fr)


def bt_ep90_pl():
    """
    Blur transform for the endpoint of the places365 distortion space
    """
    kernel_size = DISTORTION_RANGE_90['places365']['blur'][0]
    std_max_fr = DISTORTION_RANGE_90['places365']['blur'][2]
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std_max_fr)


def bt_mp_pl():
    """
    Blur transform for the midpoint of the places365 distortion space
    """
    kernel_size = DISTORTION_RANGE['places365']['blur'][0]
    std_min_fr = DISTORTION_RANGE['places365']['blur'][1]
    std_max_fr = DISTORTION_RANGE['places365']['blur'][2]
    std = (std_min_fr + std_max_fr) / 2
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bt_mp90_pl():
    """
    Blur transform for the midpoint of the places365 distortion space
    """
    kernel_size = DISTORTION_RANGE_90['places365']['blur'][0]
    std_min_fr = DISTORTION_RANGE_90['places365']['blur'][1]
    std_max_fr = DISTORTION_RANGE_90['places365']['blur'][2]
    std = (std_min_fr + std_max_fr) / 2
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def nt_fr_pl():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the sat6 nose range scaled
    by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_range = DISTORTION_RANGE['places365']['noise']
    clamp = True
    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_fr_pl_cp():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the sat6 nose range scaled
    by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """

    raise NotImplementedError('this function used wrong noise class - replaced by nt_fr_pl_cp_fixed()')

    sigma_poisson_range = DISTORTION_RANGE['coco']['noise']
    clamp = True

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_fr_pl_cp_fixed():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the sat6 nose range scaled
    by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_range = DISTORTION_RANGE['coco']['noise']
    clamp = True
    return VariablePoissonNoiseIndependentChannel(sigma_poisson_range, clamp)


def nt_0_pl():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the lower half of the
    places nose range, scaled by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor =
    input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE['places365']['noise']
    sigma_poisson_min = sigma_poisson_min_fr
    sigma_poisson_max = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    sigma_poisson_range = (sigma_poisson_min, sigma_poisson_max)
    clamp = True
    return VariablePoissonNoiseIndependentChannel(sigma_poisson_range, clamp)


def nt_1_pl():
    """
    returns a custom transform that adds zero-centered, channel-replicated Poisson noise from the upper half of the
    places nose range, scaled by 1 /255 and clamps the final output tensor to fall on [0, 1], where output_tensor =
    input_tensor + Poisson noise.

    Intended to be used in dataloader via Transforms.compose().
    """
    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE['places365']['noise']
    sigma_poisson_min = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    sigma_poisson_max = sigma_poisson_max_fr
    sigma_poisson_range = (sigma_poisson_min, sigma_poisson_max)
    clamp = True
    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_ep_pl():
    """
    Noise transform at endpoint of sat6 distortion space
    """
    __, sigma_poisson_max_fr = DISTORTION_RANGE['places365']['noise']
    clamp = True
    sigma_poisson_range = (sigma_poisson_max_fr, sigma_poisson_max_fr)

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_ep90_pl():
    """
    Noise transform at endpoint of sat6 distortion space
    """
    __, sigma_poisson_max_fr = DISTORTION_RANGE_90['places365']['noise']
    clamp = True
    sigma_poisson_range = (sigma_poisson_max_fr, sigma_poisson_max_fr)

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_mp_pl():
    """
    Noise transform at midpoint of places365 distortion space
    """
    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE['places365']['noise']
    sigma_poisson_midpoint = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    clamp = True
    sigma_poisson_range = (sigma_poisson_midpoint, sigma_poisson_midpoint)

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


def nt_mp90_pl():
    """
    Noise transform at midpoint of places365 distortion space
    """
    sigma_poisson_min_fr, sigma_poisson_max_fr = DISTORTION_RANGE_90['places365']['noise']
    sigma_poisson_midpoint = int((sigma_poisson_min_fr + sigma_poisson_max_fr) / 2)
    clamp = True
    sigma_poisson_range = (sigma_poisson_midpoint, sigma_poisson_midpoint)

    return VariablePoissonNoiseChannelReplicated(sigma_poisson_range, clamp)


tag_to_transform = {
    # sat6 transforms
    'rt_fr_s6': rt_fr_s6,
    'rt_0_s6': rt_0_s6,
    'rt_1_s6': rt_1_s6,
    'rt_ep_s6': rt_ep_s6,
    'rt_mp_s6': rt_mp_s6,

    'bt_fr_s6': bt_fr_s6,
    'bt_0_s6': bt_0_s6,
    'bt_1_s6': bt_1_s6,
    'bt_ep_s6': bt_ep_s6,
    'bt_mp_s6': bt_mp_s6,
    'bt_mp90_s6': bt_mp90_s6,

    'nt_fr_s6': nt_fr_s6,
    'nt_0_s6': nt_0_s6,
    'nt_1_s6': nt_1_s6,
    'nt_ep_s6': nt_ep_s6,
    'nt_mp_s6': nt_mp_s6,

    # places transforms
    'rt_fr_pl': rt_fr_pl,
    'rt_0_pl': rt_0_pl,
    'rt_1_pl': rt_1_pl,
    'rt_ep_pl': rt_ep_pl,
    'rt_ep90_pl': rt_ep90_pl,
    'rt_mp_pl': rt_mp_pl,
    'rt_mp90_pl': rt_mp90_pl,

    'bt_fr_pl': bt_fr_pl,
    'bt_0_pl': bt_0_pl,
    'bt_1_pl': bt_1_pl,
    'bt_ep_pl': bt_ep_pl,
    'bt_ep90_pl': bt_ep90_pl,
    'bt_mp_pl': bt_mp_pl,
    'bt_mp90_pl': bt_mp90_pl,

    'nt_fr_pl': nt_fr_pl,
    'nt_0_pl': nt_0_pl,
    'nt_1_pl': nt_1_pl,
    'nt_ep_pl': nt_ep_pl,
    'nt_ep90_pl': nt_ep90_pl,
    'nt_mp_pl': nt_mp_pl,
    'nt_mp90_pl': nt_mp90_pl,

    'rt_fr_pl_cp': rt_fr_pl_cp,
    'bt_fr_pl_cp': bt_fr_pl_cp,
    'nt_fr_pl_cp': nt_fr_pl_cp,

    'nt_fr_pl_cp_fixed': nt_fr_pl_cp_fixed,

}
