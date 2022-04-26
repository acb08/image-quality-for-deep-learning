import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from PIL import Image


RNG = np.random.default_rng()


# class AddPoissonNoise(object):
#     """
#     Adds zero-centered Poisson noise of standard deviation sigma_poisson.
#
#     sigma_poisson is intended to be specified in digital number to be added to an 8-bit image, but it is applied
#     assuming the image has already been scaled to fall on [0, 1], so the zero-centered Poisson distribution
#     is divided by 255 to match the input image scaling.
#     """
#
#     def __init__(self, sigma_poisson, clamp=True):
#         self.lambda_poisson = sigma_poisson ** 2
#         self.clamp = clamp
#
#     def __call__(self, tensor):
#         noise = (torch.poisson(torch.ones(tensor.shape) * self.lambda_poisson).float() - self.lambda_poisson) / 255
#         noisy_tensor = tensor + noise
#         if self.clamp:
#             noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
#         return noisy_tensor
#
#
# class AddVariablePoissonNoise(object):
#     """
#     Selects Poisson noise standard deviation on sigma_poisson_range with uniform probability
#     inclusive of the endpoints.
#
#     generates zero-centered poisson noise with mean selected as described above, divides the noise by 255,
#     adds to input tensor and clamps the output to fall on [0, 1]
#
#
#
#     """
#
#     def __init__(self, sigma_poisson_range, clamp=True):
#         self.low = sigma_poisson_range[0] ** 2
#         self.high = sigma_poisson_range[1] ** 2 + 1  # add one because torch.randint excludes right endpoint
#         self.clamp = clamp
#
#     def __call__(self, tensor):
#         lambda_poisson = torch.randint(self.low, self.high, (1,))
#         noise = (torch.poisson(torch.ones(tensor.shape) * lambda_poisson).float() - lambda_poisson) / 255
#         noisy_tensor = tensor + noise
#         if self.clamp:
#             noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
#         return noisy_tensor


class VariablePoissonNoiseChannelReplicated(object):
    """
    Selects Poisson noise standard deviation on sigma_poisson_range with uniform probability
    inclusive of the endpoints.

    generates zero-centered poisson noise with mean selected as described above, divides the noise by 255,
    adds to input tensor and clamps the output to fall on [0, 1]

    Copies the noise across input image channels so that a panchromatic image where the RGB channels are identical
    has the same noise added to each channel.

    """

    def __init__(self, sigma_poisson_range, clamp=True):
        self.low = sigma_poisson_range[0] ** 2
        self.high = sigma_poisson_range[1] ** 2 + 1  # add one because torch.randint excludes right endpoint
        self.clamp = clamp

    def __call__(self, tensor):
        batch_size, channels, height, width = tensor.shape
        noise_shape = (batch_size, 1, height, width)

        lambda_poisson = torch.randint(self.low, self.high, (1,))
        noise = (torch.poisson(torch.ones(noise_shape) * lambda_poisson).float() - lambda_poisson) / 255
        channel_replicated_noise = noise.repeat(1, channels, 1, 1)
        noisy_tensor = tensor + channel_replicated_noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class VariableImageResize(object):
    """
    Builds a transform bank to re-size images and provides a method for randomly selecting the size of the
    transform to be used before being called.
    """

    def __init__(self, sizes, interpolation_mode='bilinear', antialias=False):
        self.sizes = sizes
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.transform_bank = self.build_transform_bank()
        self.size_keys = list(self.transform_bank.keys())

    def build_transform_bank(self):
        transform_bank = {}
        for size in self.sizes:
            size = int(size)
            if self.interpolation_mode == 'bilinear':
                new_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR,
                                       antialias=self.antialias),
                     transforms.ToPILImage()]
                )
            elif self.interpolation_mode == 'bicubic':
                new_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC,
                                       antialias=self.antialias),
                     transforms.ToPILImage()]
                )
            else:
                raise Exception('Invalid interpolation_mode')
            transform_bank[size] = new_transform
        print('build_transform_bank called')
        return transform_bank

    def get_size_key(self):
        """
        randomly returns one of the class's size keys, which can then be used in the __call__() method. Allows the user
        to know the resolution ahead of time for things like initializing arrays of the correct shape.
        """
        size_key = random.choice(self.size_keys)
        return size_key

    def __call__(self, image, size_key, dtype=np.uint8):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        transform_use = self.transform_bank[size_key]
        image = transform_use(image)
        image = np.asarray(image, dtype=dtype)
        return image


class VariableResolution(object):

    """
    Randomly selects a resize transform per the sizes argument passed in the __init__() method
    """

    def __init__(self, sizes, interpolation_mode='bilinear', antialias=False):
        self.sizes = sizes
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias
        self.transform_bank = self.build_transform_bank()

    def build_transform_bank(self):
        transform_bank = []
        for size in self.sizes:
            size = int(size)
            if self.interpolation_mode == 'bilinear':
                new_transform = transforms.Resize(size,
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=self.antialias)
            elif self.interpolation_mode == 'bicubic':
                new_transform = transforms.Resize(size,
                                                  interpolation=transforms.InterpolationMode.BICUBIC,
                                                  antialias=self.antialias)
            else:
                raise Exception('Invalid interpolation_mode')
            transform_bank.append(new_transform)
        print('build_transform_bank called')
        return transform_bank

    def __call__(self, tensor):
        transform_use = random.choice(self.transform_bank)
        return transform_use(tensor)


def _add_zero_centered_channel_replicated_poisson_noise(img, lambda_poisson):

    img = np.asarray(img)
    h, w, c = img.shape
    noise = RNG.poisson(lambda_poisson, size=(h, w)) - lambda_poisson
    noise_replicated = np.stack(c * [noise], axis=2)
    img_out = img + noise_replicated
    img_out = np.clip(img_out, 0, 255)

    return img_out


# n_scan did not replicate noise across channels
# def n_scan(img):
#     """
#     :param img: image array, values on [0, 255]
#     :return: image + zero centered Poisson noise, where the resulting image is clamped to fall on [0, 255]
#
#     """
#
#     img = np.asarray(img)
#     shape = img.shape
#
#     sigma_poisson = np.random.randint(0, 26)  # add 1 to target distribution, high is "one above the highest integer
#     to be drawn from the target distribution
#     lambda_poisson = sigma_poisson ** 2
#     noise = RNG.poisson(lambda_poisson, size=shape) - lambda_poisson
#     img_out = img + noise
#     img_out = np.clip(img_out, 0, 255)
#
#     return img_out, 'lambda_poisson', lambda_poisson


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


def r_scan_pl():
    """
    Initializes and returns a VariableImageResize instance designed to re-size SAT6 images
    randomly between 25% and 100% of original images size. Intended for use in dataset distortion (as opposed to in
    a dataloader)
    """

    max_size = 256
    res_fractions = np.linspace(0.15, 1, num=20)
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
    # 'n_scan': n_scan, # n_scan did not replicate noise over image channels
    'n_scan_v2': n_scan_v2,
    'n_scan_v3': n_scan_v3,

    'r_scan_pl': r_scan_pl  # places
}

tag_to_transform = {}
