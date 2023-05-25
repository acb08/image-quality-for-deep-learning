import random

import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from src.pre_processing.distortion_tools import image_to_electrons, electrons_to_image


class VariableResolution:

    """
    Randomly selects a resize transform per the sizes argument passed in the __init__() method. Intended for use
    in a dataloader
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


class VariableCOCOResize:
    """
    Builds a transform bank to re-size images and provides a method for randomly selecting the size of the
    transform to be used before being called.
    """

    def __init__(self, interpolation_mode='bilinear', antialias=False):
        # self.resolution_fractions = resolution_fractions
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias

    def __call__(self, image, res_frac, dtype=np.uint8):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        min_dimension = min(np.shape(image)[:2])
        new_size = int(res_frac * min_dimension)

        if self.interpolation_mode == 'bilinear':
            transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR,
                                           antialias=self.antialias),
                         transforms.ToPILImage()]
                    )
        elif self.interpolation_mode == 'bicubic':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BICUBIC,
                                   antialias=self.antialias),
                 transforms.ToPILImage()]
            )
        else:
            raise Exception('Invalid interpolation_mode')

        image = transform(image)

        return image


class VariableImageResize:
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


class VariablePoissonNoiseChannelReplicated:
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
        # batch_size, channels, height, width = tensor.shape
        channels, height, width = tensor.shape
        noise_shape = (1, height, width)

        lambda_poisson = torch.randint(self.low, self.high, (1,))
        noise = (torch.poisson(torch.ones(noise_shape) * lambda_poisson).float() - lambda_poisson) / 255
        # channel_replicated_noise = noise.repeat(1, channels, 1, 1)
        channel_replicated_noise = noise.repeat(channels, 1, 1)
        noisy_tensor = tensor + channel_replicated_noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class PseudoSensor:
    """
    Scales signal and noise properties of an input image based on its associated blur and resolution metadata
    """
    def __init__(self, read_noise_value, input_image_well_depth):
        self._dc_fraction_estimator = None
        self.read_noise_value = read_noise_value
        self.input_image_well_depth = input_image_well_depth

    def convert_scale_electrons(self, image, res_frac):
        electrons = image_to_electrons(image, self.input_image_well_depth)
        scaled_electrons = self.rescale(value=electrons, res_frac=res_frac)
        return scaled_electrons

    def apply_read_noise(self, electrons):
        noise = np.random.randn(*np.shape(electrons)) * self.read_noise_value
        noise = np.round(noise, 0)
        electrons = electrons + noise
        return electrons

    def output_image_well_depth(self, res_frac):
        """
        Scales well depth by the ratio of output pixels area to input pixel area
        """
        return self.rescale(value=self.input_image_well_depth, res_frac=res_frac)

    @staticmethod
    def apply_poisson_distribution(electrons):
        return np.random.poisson(electrons)

    @staticmethod
    def rescale(value, res_frac):
        scale = (1 / res_frac) ** 2
        return scale * value

    @staticmethod
    def clip_electrons(electrons, well_depth):
        return np.clip(electrons, 0, well_depth)

    def __call__(self, image, res_frac):

        output_well_depth = self.output_image_well_depth(res_frac=res_frac)

        electrons = self.convert_scale_electrons(image, res_frac)
        electrons = self.apply_poisson_distribution(electrons)
        electrons = self.apply_read_noise(electrons)
        electrons = self.clip_electrons(electrons, well_depth=output_well_depth)

        image = electrons_to_image(electrons=electrons, well_depth=output_well_depth)

        return image, self.read_noise_value









