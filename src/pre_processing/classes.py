import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from src.pre_processing.distortion_tools import image_to_electrons, electrons_to_image


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

        return image, None, 'noise', self.read_noise_value


class PseudoSensorVariedSNR:
    """
    Simulates imaging in varied SNR regimes with variable resolution / pixel size. Globally scales input signal,
    applies fixed read noise and dark current that is inversely proportional to signal scaling (i.e. the dark current
    increases to account for longer integration times when signal is lower).  Scales signal up with when resolution
    decreases / pixel size increases, with signal scaling that is inversely proportional to the square of resolution
    fraction / directly proportional to pixel area ratio.
    """
    def __init__(self, signal_fraction, read_noise, input_image_well_depth, baseline_dark_current):
        """

        :param signal_fraction: float, scale factor input signal
        :param read_noise:
        :param input_image_well_depth:
        :param baseline_dark_current:
        """
        self.signal_fraction = signal_fraction
        self.read_noise = read_noise
        self.input_image_well_depth = input_image_well_depth

        self.min_output_well_depth = self.set_min_output_well_depth()
        self._baseline_dark_current = baseline_dark_current

        self.avg_dark_current = self.get_dark_current()

    def set_min_output_well_depth(self):
        """
        Assumes that all input images all come from sensors with the same well depth.  To avoid downstream
        normalization issues from images that do not fill the full dynamic range, the PseudoSensorVariedSNR class
        scales well depth with signal attenuation, so that input and output images of differing SNR still have
        similar dynamic ranges.
        """
        return self.signal_fraction * self.input_image_well_depth

    def get_dark_current(self):
        """
        Assumes that the best SNR scenario corresponds to a minimum integration time that yields a low baseline dark
        current.  When the signal level is attenuated, the integration time and dark current are assumed to increase
        inversely.
        """
        return self._baseline_dark_current / self.signal_fraction

    def apply_read_noise(self, electrons):
        noise = np.random.randn(*np.shape(electrons)) * self.read_noise
        noise = np.round(noise, 0)
        electrons = electrons + noise
        return electrons

    def apply_dark_current(self, electrons):
        dark_electrons = RNG.poisson(lam=self.avg_dark_current, size=np.shape(electrons))
        return electrons + dark_electrons

    def attenuate(self, electrons):
        return self.signal_fraction * electrons

    @staticmethod
    def pix_rescale(value, res_frac):
        scale = (1 / res_frac) ** 2
        return scale * value

    @staticmethod
    def apply_poisson_distribution(signal):
        return RNG.poisson(lam=signal)

    @staticmethod
    def clip_electrons(electrons, well_depth):
        return np.clip(electrons, 0, well_depth)

    @staticmethod
    def quantization_noise(well_depth):
        quantization_step = well_depth / 2 ** 8
        quantization_variance = quantization_step ** 2 / 12
        return np.std(quantization_variance)

    def subtract_dark_offset(self, electrons):
        """
        Effectively zero-centers the dark current Poisson distribution. Used to avoid normalization issues that could
        arise from significant dc bias in the images
        """
        return electrons - self.avg_dark_current

    def output_image_well_depth(self, res_frac):
        """
        Scales well depth by the ratio of output pixel area to input pixel area
        """
        return self.pix_rescale(value=self.min_output_well_depth, res_frac=res_frac)

    def __call__(self, image, res_frac):

        output_well_depth = self.output_image_well_depth(res_frac=res_frac)

        electrons = image_to_electrons(image, self.input_image_well_depth)
        initial_mean = np.mean(electrons)
        electrons = self.attenuate(electrons)
        attenuated_mean = np.mean(electrons)

        electrons = self.pix_rescale(value=electrons, res_frac=res_frac)

        approx_signal = np.max(electrons) - np.min(electrons)  # very rudimentary

        noisy_electrons = self.apply_poisson_distribution(electrons)
        noisy_electrons = self.apply_read_noise(noisy_electrons)
        noisy_electrons = self.apply_dark_current(noisy_electrons)
        noisy_electrons = self.subtract_dark_offset(noisy_electrons)
        noisy_electrons = self.clip_electrons(noisy_electrons, well_depth=output_well_depth)

        electron_noise = np.std(noisy_electrons - electrons)
        quantization_noise = self.quantization_noise(well_depth=output_well_depth)
        approx_noise = np.sqrt(electron_noise ** 2 + quantization_noise ** 2)

        approx_snr = approx_signal / approx_noise

        output_image = electrons_to_image(electrons=noisy_electrons, well_depth=output_well_depth)

        print(round(float(initial_mean), 0), round(float(attenuated_mean), 0), round(attenuated_mean / initial_mean, 2))
        print(approx_snr, '\n')

        return output_image, approx_snr, 'noise', approx_noise







