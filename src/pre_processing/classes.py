import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from src.pre_processing.distortion_tools import image_to_electrons, electrons_to_image, relative_aperture
from src.utils.definitions import BASELINE_HIGH_SIGNAL_WELL_DEPTH, BASELINE_SIGMA_BLUR, BASELINE_READ_NOISE, \
    BASELINE_DARK_CURRENT


RNG = np.random.default_rng()


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


class VariablePoissonNoiseIndependentChannel:
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

        noise_shape = tensor.shape
        lambda_poisson = torch.randint(self.low, self.high, (1,))
        noise = (torch.poisson(torch.ones(noise_shape) * lambda_poisson).float() - lambda_poisson) / 255
        noisy_tensor = tensor + noise

        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class PseudoSensorFixedWell:
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

        return image, None, 'noise', self.read_noise_value


class PseudoSensor:
    """
    Simulates imaging in varied SNR regimes with variable resolution / pixel size. Globally scales input signal,
    applies fixed read noise and dark current that is inversely proportional to signal scaling (i.e. the dark current
    increases to account for longer integration times when signal is lower).  Scales signal up with when resolution
    decreases / pixel size increases, with signal scaling that is inversely proportional to the square of resolution
    fraction / directly proportional to pixel area ratio.
    """
    def __init__(self, signal_fraction):
        """

        :param signal_fraction: float, scale factor input signal
        """

        self.signal_fraction = signal_fraction
        self.read_noise = BASELINE_READ_NOISE
        self.well_depth = np.sqrt(self.signal_fraction) * BASELINE_HIGH_SIGNAL_WELL_DEPTH
        self._assumed_input_well_depth = BASELINE_HIGH_SIGNAL_WELL_DEPTH  # diagnostic use only

    def apply_read_noise(self, electrons):
        noise = np.random.randn(*np.shape(electrons)) * self.read_noise
        noise = np.round(noise, 0)
        electrons = electrons + noise
        return electrons

    @staticmethod
    def apply_dark_current(electrons, mean_dark_count):
        dark_electrons = RNG.poisson(lam=mean_dark_count, size=np.shape(electrons))
        return electrons + dark_electrons

    def get_mean_dark_count(self, sigma_blur):
        return sigma_blur ** 2 / np.sqrt(self.signal_fraction) * BASELINE_DARK_CURRENT

    def attenuate(self, electrons):
        return self.signal_fraction * electrons

    @staticmethod
    def scale_for_pix_size(value, res_frac):
        scale = (1 / res_frac) ** 2
        return scale * value

    @staticmethod
    def scale_for_relative_aperture_size(value, relative_aperture_size):
        """
        Scales signal according to relative aperture diameter.  Assumes a baseline diameter of 1, with signal varying
        with the square of aperture size (i.e. diameter)
        """
        scale = relative_aperture_size ** 2
        return scale * value

    @staticmethod
    def apply_poisson_distribution(signal):
        return RNG.poisson(lam=signal)

    @staticmethod
    def clip_electrons(electrons, well_depth):
        return np.clip(electrons, 0, well_depth)

    @staticmethod
    def quantization_noise(well_depth):
        """
        Estimates quantization noise in units of electrons
        """
        quantization_step = well_depth / 2 ** 8
        quantization_variance = quantization_step ** 2 / 12
        return np.sqrt(quantization_variance)

    @staticmethod
    def subtract_dark_offset(electrons, mean_dark_count):
        """
        Effectively zero-centers the dark current Poisson distribution. Used to avoid normalization issues that could
        arise from significant dc bias in the images
        """
        return electrons - mean_dark_count

    def output_image_well_depth(self, res_frac, relative_aperture_size):
        """
        Scales well depth by the ratio of output pixel area to input pixel area
        """
        well_depth = self.scale_for_pix_size(value=self.well_depth, res_frac=res_frac)
        well_depth = self.scale_for_relative_aperture_size(
            value=well_depth, relative_aperture_size=relative_aperture_size)

        assert np.abs(well_depth == self.well_depth * (relative_aperture_size / res_frac) ** 2) < 10

        return well_depth

    @staticmethod
    def electron_noise_to_dn(noise_electrons, well_depth):
        normed_noise = noise_electrons / well_depth
        assert 0 <= normed_noise <= 1
        noise_dn = int((2 ** 8 - 1) * normed_noise)
        return noise_dn

    def __call__(self, image, res_frac, sigma_blur, signal_est_method='range', log_file=None, eps=0.01):

        pre_noise_electrons = image_to_electrons(image, well_depth=self.well_depth)

        # _diagnostic _only
        _relative_aperture_size = relative_aperture(sigma_blur=sigma_blur)

        signal_range_electrons = np.max(pre_noise_electrons) - np.min(pre_noise_electrons)  # very rudimentary
        signal_mean_electrons = np.mean(pre_noise_electrons)

        mean_dark_count = self.get_mean_dark_count(sigma_blur=sigma_blur)

        noisy_electrons = self.apply_poisson_distribution(pre_noise_electrons)
        noisy_electrons = self.apply_read_noise(noisy_electrons)
        noisy_electrons = self.apply_dark_current(noisy_electrons, mean_dark_count=mean_dark_count)
        noisy_electrons = self.subtract_dark_offset(noisy_electrons, mean_dark_count=mean_dark_count)
        noisy_electrons = self.clip_electrons(noisy_electrons, well_depth=self.well_depth)

        measured_electron_noise = np.std(noisy_electrons - pre_noise_electrons)
        quantization_noise = self.quantization_noise(well_depth=self.well_depth)  # in units of pre_noise_electrons
        estimated_post_adc_noise_electrons = np.sqrt(measured_electron_noise ** 2 + quantization_noise ** 2)

        signal_range_snr = signal_range_electrons / estimated_post_adc_noise_electrons
        signal_mean_snr = signal_mean_electrons / estimated_post_adc_noise_electrons

        if signal_est_method == 'range':
            est_snr = signal_range_snr
        elif signal_est_method == 'mean':
            est_snr = signal_mean_snr
        else:
            raise ValueError("signal_est_method must be either 'range' or 'mean'")

        analytical_electron_noise = np.sqrt(np.mean(pre_noise_electrons) + mean_dark_count + BASELINE_READ_NOISE ** 2)

        diagnostic_data = {
            'sigma_blur': sigma_blur,
            'mean_dark_count': mean_dark_count,
            'dark_est_std': np.sqrt(mean_dark_count),
            'mean_pre_noise_electrons': np.mean(pre_noise_electrons),
            'est_shot_noise': np.sqrt(np.mean(pre_noise_electrons)),
            'analytical_noise_electrons': analytical_electron_noise,
            'read_noise': BASELINE_READ_NOISE,
            'measured_electron_noise': measured_electron_noise,
            'estimated_post_adc_noise_electrons': estimated_post_adc_noise_electrons,
            'well_depth': self.well_depth,
            'relative_aperture_size': _relative_aperture_size
        }
        
        output_image = electrons_to_image(electrons=noisy_electrons, well_depth=self.well_depth)

        return output_image, est_snr, 'snr', diagnostic_data
