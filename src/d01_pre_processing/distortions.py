import numpy as np
from torchvision import transforms

from src.d00_utils.classes import VariableImageResize

RNG = np.random.default_rng()


def _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson):

    img = np.asarray(img)
    h, w, c = img.shape
    noise = RNG.poisson(lambda_poisson, size=(h, w)) - lambda_poisson
    noise_replicated = np.stack(c * [noise], axis=2)
    img_out = img + noise_replicated
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
    randomly between 25% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    max_size = 256
    res_fractions = np.linspace(0.1, 1, num=20)
    sizes = [int(res_frac * max_size) for res_frac in res_fractions]
    transform = VariableImageResize(sizes)

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

    'b_fr_s6': b_fr_s6,  # sat6
    'b_fr_pl': b_fr_pl,  # places

    'n_fr_s6': n_fr,  # sat6 (same transform for places and sat6)
    'n_fr_pl': n_fr,  # places (same transform for places and sat6)

}
