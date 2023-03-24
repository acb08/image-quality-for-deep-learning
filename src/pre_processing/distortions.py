import random
from PIL import Image
import numpy as np
from torchvision import transforms
from src.utils.definitions import DISTORTION_RANGE, NATIVE_RESOLUTION, DISTORTION_RANGE_90
from src.pre_processing.classes import VariableCOCOResize, VariableImageResize

RNG = np.random.default_rng()


def _get_kernel_size(std):
    return 8 * max(int(np.round(std, 0)), 1) + 1


def _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson):

    img = np.asarray(img)
    h, w, c = img.shape
    noise = RNG.poisson(lambda_poisson, size=(h, w)) - lambda_poisson
    noise_replicated = np.stack(c * [noise], axis=2)
    img_out = img + noise_replicated
    img_out = np.clip(img_out, 0, 255)

    return img_out


def _add_zero_centered_poisson_noise(img, lambda_poisson):

    img = np.asarray(img)

    noise = RNG.poisson(lambda_poisson, size=img.shape) - lambda_poisson
    img_out = img + noise
    img_out = np.clip(img_out, 0, 255)

    return img_out


def n_scan_v2(img):
    """
    Adds zero-centered, channel-replicated Poisson noise up to 25 DN.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """

    sigma_max = 25
    sigma_poisson = np.random.randint(0, sigma_max + 1)  # add 1 to target distribution, high is one above the highest
    # integer to be drawn from the target distribution
    lambda_poisson = sigma_poisson ** 2
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_scan_v3(img):
    """
    Adds zero-centered, channel-replicated Poisson noise up to 50 DN.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_max = 50
    step = 2
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_fr(img):
    """
    Full range noise distortion for both sat6 and places365.

    Adds zero-centered, channel-replicated Poisson noise up to 50 DN.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_max = 50
    step = 2
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_fr90_pl(img):
    """
    Full range noise distortion for both sat6 and places365.

    Adds zero-centered, channel-replicated Poisson noise up to 50 DN.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_min, sigma_max = DISTORTION_RANGE_90['places365']['noise']
    step = 2
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


# def n_ep(img):
#     """
#     End point noise distortion for both sat6 and places365.
#
#     Adds 50 DN zero-centered, channel-replicated Poisson noise.
#
#     :param img: image array, values on [0, 255]
#     :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]
#
#     """
#     sigma_poisson = 50
#     lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
#     img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)
#
#     return img_out, 'lambda_poisson', lambda_poisson


def n_ep_pl(img):
    """
    End point noise distortion places365.

    Adds 50 DN zero-centered, channel-replicated Poisson noise.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_poisson_min, sigma_poisson_max = DISTORTION_RANGE['places365']['noise']
    lambda_poisson = int(sigma_poisson_max ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_ep90_pl(img):
    """
    End point 90 noise distortion places365.

    Adds 44 DN zero-centered, channel-replicated Poisson noise.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_poisson_min, sigma_poisson_max = DISTORTION_RANGE_90['places365']['noise']
    lambda_poisson = int(sigma_poisson_max ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_ep_s6(img):
    """
    End point noise distortion for sat6.

    Adds 50 DN zero-centered, channel-replicated Poisson noise.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_poisson_min, sigma_poisson_max = DISTORTION_RANGE['sat6']['noise']
    lambda_poisson = int(sigma_poisson_max ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_mp_s6(img):
    """
    Midpoint noise distortion for sat6.

    Adds 25 DN zero-centered, channel-replicated Poisson noise.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_poisson_min, sigma_poisson_max = DISTORTION_RANGE['sat6']['noise']
    sigma_poisson = (sigma_poisson_min + sigma_poisson_max) / 2
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_mp_pl(img):
    """
    Midpoint noise distortion for sat6.

    Adds 25 DN zero-centered, channel-replicated Poisson noise.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_poisson_min, sigma_poisson_max = DISTORTION_RANGE['places365']['noise']
    sigma_poisson = (sigma_poisson_min + sigma_poisson_max) / 2
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def n_mp90_pl(img):
    """
    Midpoint 90 noise distortion for places365.

    Adds 22 DN zero-centered, channel-replicated Poisson noise.

    :param img: image array, values on [0, 255]
    :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]

    """
    sigma_poisson_min, sigma_poisson_max = DISTORTION_RANGE_90['places365']['noise']
    sigma_poisson = (sigma_poisson_min + sigma_poisson_max) / 2
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson)

    return img_out, 'lambda_poisson', lambda_poisson


def b_scan(img):

    kernel_size = 23
    sigma_range = np.linspace(0.1, 4, num=21, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_scan_v2(img):

    kernel_size = 17
    sigma_range = np.linspace(0.1, 2.5, num=21, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_fr_s6(img):

    kernel_size = 11
    sigma_range = np.linspace(0.1, 1.5, num=21, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_fr90_s6(img):

    kernel_size, sigma_min, sigma_max = DISTORTION_RANGE_90['sat6']['blur']
    sigma_range = np.linspace(sigma_min, sigma_max, num=15, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_ep_s6(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE['sat6']['blur']
    std = max_blur

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_mp_s6(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE['sat6']['blur']
    std = (min_blur + max_blur) / 2

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_mp90_s6(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE_90['sat6']['blur']
    std = (min_blur + max_blur) / 2

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_scan_v3(img):

    kernel_size = 31
    sigma_range = np.linspace(0.1, 5, num=21, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_fr_pl(img):

    kernel_size = 31
    sigma_range = np.linspace(0.1, 5, num=21, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_fr90_pl(img):

    kernel_size, sigma_min, sigma_max = DISTORTION_RANGE_90['places365']['blur']
    sigma_range = np.linspace(sigma_min, sigma_max, num=15, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_ep_pl(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE['places365']['blur']
    std = max_blur

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_ep90_pl(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE_90['places365']['blur']
    std = max_blur

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_mp_pl(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE['places365']['blur']
    std = (min_blur + max_blur) / 2

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_mp90_pl(img):

    kernel_size, min_blur, max_blur = DISTORTION_RANGE_90['places365']['blur']
    std = (min_blur + max_blur) / 2

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b_test_coco(img):

    kernel_size, sigma_min, sigma_max = 15, 0.5, 4
    sigma_range = np.linspace(sigma_min, sigma_max, num=15, endpoint=True)
    std = np.random.choice(sigma_range)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), None, 'blur', std


def b0_coco(img):

    kernel_size, sigma_min, sigma_max = 15, 0.5, 4
    sigma_range = np.linspace(sigma_min, sigma_max, num=7, endpoint=True)
    std = np.random.choice(sigma_range)
    # img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), None, 'blur', std


def b_scan_coco(img):

    sigma_min, sigma_max = 0.5, 5
    sigma_range = np.linspace(sigma_min, sigma_max, num=11, endpoint=True)
    std = np.random.choice(sigma_range)

    kernel_size = int(5 * std)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), None, 'blur', std


def b_scan_coco_v2(img):

    sigma_min, sigma_max = 0.5, 10
    sigma_range = np.linspace(sigma_min, sigma_max, num=11, endpoint=True)
    std = np.random.choice(sigma_range)

    kernel_size = int(5 * std)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), None, 'blur', std


def b_fr_tr_coco(img):

    sigma_min, sigma_max = 0.1, 5
    sigma_range = np.linspace(sigma_min, sigma_max, num=20, endpoint=True)
    std = np.random.choice(sigma_range)

    kernel_size = int(5 * std)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), None, 'blur', std


def b_fr90_coco(img):

    sigma_range = DISTORTION_RANGE_90['coco']['blur']
    std = np.random.choice(sigma_range)

    kernel_size = _get_kernel_size(std)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), None, 'blur', std


def no_op_coco(img):
    """
    Debugging function to check out data pipeline
    """
    return np.asarray(img, dtype=np.uint8), None, 'no_dist', 0


def n0_coco(img):
    """
    Debugging function to check out data pipeline
    """

    sigma_min, sigma_max = 0, 30
    step = 5
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_poisson_noise(img, lambda_poisson)

    img_out = np.asarray(img_out, dtype=np.uint8)

    return img_out, None, 'noise', lambda_poisson


def n_scan_coco(img):
    """
    Debugging function to check out data pipeline
    """

    sigma_min, sigma_max = 0, 50
    step = 5
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_poisson_noise(img, lambda_poisson)

    img_out = np.asarray(img_out, dtype=np.uint8)

    return img_out, None, 'noise', lambda_poisson


def n_scan_coco_v2(img):
    """
    Debugging function to check out data pipeline
    """

    sigma_min, sigma_max = 0, 100
    step = 10
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_poisson_noise(img, lambda_poisson)

    img_out = np.asarray(img_out, dtype=np.uint8)

    return img_out, None, 'noise', lambda_poisson


def n_fr_tr_coco(img):
    """
    Debugging function to check out data pipeline
    """

    sigma_min, sigma_max = 0, 80
    step = 5
    sigma_vals = step * np.arange(int(sigma_max / step) + 1)
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_poisson_noise(img, lambda_poisson)

    img_out = np.asarray(img_out, dtype=np.uint8)

    return img_out, None, 'noise', lambda_poisson


def n_fr90_coco(img):
    """
    Debugging function to check out data pipeline
    """

    sigma_vals = DISTORTION_RANGE_90['coco']['noise']
    sigma_poisson = np.random.choice(sigma_vals)
    lambda_poisson = int(sigma_poisson ** 2)  # convert from np.int64 to regular int for json serialization
    img_out = _add_zero_centered_poisson_noise(img, lambda_poisson)

    img_out = np.asarray(img_out, dtype=np.uint8)

    return img_out, None, 'noise', lambda_poisson


def r0_coco(img):

    res_frac = random.choice([0.4, 0.6, 0.7, 0.8, 0.9, 1])
    img_out = VariableCOCOResize()(img, res_frac)

    return img_out, None, 'res', res_frac


def r_scan_coco(img):

    res_frac = random.choice([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1])
    img_out = VariableCOCOResize()(img, res_frac)

    return img_out, None, 'res', res_frac


def r_scan_coco_v2(img):

    res_frac = random.choice([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1])
    img_out = VariableCOCOResize()(img, res_frac)

    return img_out, None, 'res', res_frac


def r_fr_tr_coco(img):

    res_fractions = np.linspace(0.2, 1, num=20)
    res_frac = random.choice(res_fractions)
    img_out = VariableCOCOResize()(img, res_frac)

    return img_out, None, 'res', res_frac


def r_fr90_coco(img):

    res_fractions = DISTORTION_RANGE_90['coco']['res']
    res_frac = random.choice(res_fractions)
    img_out = VariableCOCOResize()(img, res_frac)

    return img_out, None, 'res', res_frac


def r_no_change_coco(img):
    """
    Debug function that does not change the image
    """
    res_frac = random.choice([1])
    img_out = VariableCOCOResize()(img, res_frac)

    return img_out, None, 'res', res_frac


def r_scan():
    """
    Initializes and returns a VariableImageResize instance designed to re-size SAT6 images
    randomly between 25% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    min_size = 7
    max_size = 28
    sizes = list(np.arange(min_size, max_size + 1))
    transform = VariableImageResize(sizes)

    return transform


def r_scan_v2():
    """
    Initializes and returns a VariableImageResize instance designed to re-size SAT6 images
    randomly between 25% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    min_size = 4
    max_size = 28
    sizes = list(np.arange(min_size, max_size + 1))
    transform = VariableImageResize(sizes)

    return transform


def r_fr_s6():
    """
    Initializes and returns a VariableImageResize instance designed to re-size SAT6 images
    randomly between 25% and 100% of original images size.

    Intended for use in dataset distortion (as opposed to in
    a dataloader). Distorts images across full distortion range for SAT6.
    """

    min_size = 7
    max_size = 28
    sizes = list(np.arange(min_size, max_size + 1))
    transform = VariableImageResize(sizes)

    return transform


def r_ep_s6():
    """
    Initializes and returns a VariableImageResize instance designed to re-size SAT6 images
    to 10% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    sizes = [7]
    transform = VariableImageResize(sizes, interpolation_mode='bilinear', antialias=False)

    return transform


def r_mp_s6():

    if NATIVE_RESOLUTION != 28:
        raise Exception('mismatch between max size and native resolution in project config')

    min_size, max_size = DISTORTION_RANGE['sat6']['res']
    size = int((min_size + max_size) / 2)
    sizes = [size]
    transform = VariableImageResize(sizes, interpolation_mode='bilinear', antialias=False)

    return transform


def r_scan_pl():
    """
    Initializes and returns a VariableImageResize instance designed to re-size Places365 images
    randomly between 25% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    max_size = 256
    res_fractions = np.linspace(0.15, 1, num=20)
    sizes = [int(res_frac * max_size) for res_frac in res_fractions]
    transform = VariableImageResize(sizes)

    return transform


def r_scan_plv2():
    """
    Initializes and returns a VariableImageResize instance designed to re-size Places365 images
    randomly between 25% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    max_size = 256
    res_fractions = np.linspace(0.1, 1, num=20)
    sizes = [int(res_frac * max_size) for res_frac in res_fractions]
    transform = VariableImageResize(sizes)

    return transform


def r_fr_pl():
    """
    Initializes and returns a VariableImageResize instance designed to re-size Places365 images
    randomly between 10% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    max_size = 256
    res_fractions = np.linspace(0.1, 1, num=20)
    sizes = [int(res_frac * max_size) for res_frac in res_fractions]
    transform = VariableImageResize(sizes)

    return transform


def r_fr90_pl():
    """
    Initializes and returns a VariableImageResize instance designed to re-size Places365 images across the "90%"
    distortion band's resolution range. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    max_size = 256
    min_resolution, max_resolution = DISTORTION_RANGE_90['places365']['res']
    res_fractions = np.linspace(min_resolution, max_resolution, num=16)
    sizes = [int(res_frac * max_size) for res_frac in res_fractions]
    transform = VariableImageResize(sizes)

    return transform


def r_ep_pl():
    """
    Initializes and returns a VariableImageResize instance designed to re-size Places365 images
    to 10% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    min_res, max_res = DISTORTION_RANGE['places365']['res']

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = [int(min_res * NATIVE_RESOLUTION)]
    transform = VariableImageResize(sizes, interpolation_mode='bilinear', antialias=False)

    return transform


def r_ep90_pl():
    """
    Initializes and returns a VariableImageResize instance designed to re-size Places365 images
    to 20% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    min_res, max_res = DISTORTION_RANGE_90['places365']['res']

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    sizes = [int(min_res * NATIVE_RESOLUTION)]
    transform = VariableImageResize(sizes, interpolation_mode='bilinear', antialias=False)

    return transform


def r_mp_pl():

    min_res, max_res = DISTORTION_RANGE['places365']['res']

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    min_size = NATIVE_RESOLUTION * min_res
    max_size = NATIVE_RESOLUTION * max_res
    size = int((min_size + max_size) / 2)
    sizes = [size]
    transform = VariableImageResize(sizes, interpolation_mode='bilinear', antialias=False)

    return transform


def r_mp90_pl():

    min_res, max_res = DISTORTION_RANGE_90['places365']['res']

    if NATIVE_RESOLUTION != 256:
        raise Exception('mismatch between max size and native resolution in project config')

    min_size = NATIVE_RESOLUTION * min_res
    max_size = NATIVE_RESOLUTION * max_res
    size = int((min_size + max_size) / 2)
    sizes = [size]
    transform = VariableImageResize(sizes, interpolation_mode='bilinear', antialias=False)

    return transform


tag_to_image_distortion = {

    # _scan transforms intended for finding the point along each distortion axis at which performance of a
    # pre-trained model drops to chance
    'r_scan': r_scan,  # sat6
    'r_scan_v2': r_scan_v2,  # sat6
    'b_scan': b_scan,
    'b_scan_v2': b_scan_v2,
    'b_scan_v3': b_scan_v3,

    # 'n_scan': n_scan, # n_scan did not replicate noise over image channels
    'n_scan_v2': n_scan_v2,
    'n_scan_v3': n_scan_v3,

    'r_scan_pl': r_scan_pl,  # places
    'r_scan_plv2': r_scan_plv2,

    'r_fr_s6': r_fr_s6,  # sat6
    'r_fr_pl': r_fr_pl,  # places
    'r_fr90_pl': r_fr90_pl,
    'r_ep_s6': r_ep_s6,  # sat6
    'r_ep_pl': r_ep_pl,  # places
    'r_ep90_pl': r_ep90_pl,
    'r_mp_s6': r_mp_s6,
    'r_mp_pl': r_mp_pl,
    'r_mp90_pl': r_mp90_pl,

    'b_fr_s6': b_fr_s6,  # sat6
    'b_fr_pl': b_fr_pl,  # places
    'b_fr90_s6': b_fr90_s6,
    'b_fr90_pl': b_fr90_pl,
    'b_ep_s6': b_ep_s6,
    'b_ep_pl': b_ep_pl,
    'b_ep90_pl': b_ep90_pl,
    'b_mp_s6': b_mp_s6,
    'b_mp90_s6': b_mp90_s6,
    'b_mp_pl': b_mp_pl,
    'b_mp90_pl': b_mp90_pl,

    'n_fr_s6': n_fr,  # sat6 (same transform for places and sat6)
    'n_fr_pl': n_fr,  # places (same transform for places and sat6)
    'n_fr90_pl': n_fr90_pl,
    'n_ep_s6': n_ep_s6,  # sat6 (same transform for places and sat6)
    'n_ep_pl': n_ep_pl,  # places (same transform for places and sat6)
    'n_ep90_pl': n_ep90_pl,
    'n_mp_s6': n_mp_s6,
    'n_mp_pl': n_mp_pl,
    'n_mp90_pl': n_mp90_pl
}


coco_tag_to_image_distortions = {  # coco distortion functions return distortion_type_flag
    'b_test_coco': b_test_coco,
    'no_op_coco': no_op_coco,

    'b0_coco': b0_coco,
    'n0_coco': n0_coco,
    'r0_coco': r0_coco,
    'r_no_change_coco': r_no_change_coco,

    'r_scan_coco': r_scan_coco,
    'b_scan_coco': b_scan_coco,
    'n_scan_coco': n_scan_coco,

    'r_scan_coco_v2': r_scan_coco_v2,
    'b_scan_coco_v2': b_scan_coco_v2,
    'n_scan_coco_v2': n_scan_coco_v2,

    'r_fr_tr_coco': r_fr_tr_coco,
    'b_fr_tr_coco': b_fr_tr_coco,
    'n_fr_tr_coco': n_fr_tr_coco,

    'r_fr90_coco': r_fr90_coco,
    'b_fr90_coco': b_fr90_coco,
    'n_fr90_coco': n_fr90_coco,
}
