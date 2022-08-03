"""
Makes a very rough estimate of the noise content in a typical 8-bit image with 50% saturation in its RGB format and
after conversion to grayscale
"""

import numpy as np

RNG = np.random.default_rng()


def pre_sample_electrons(image_size, saturation_frac, well_depth, dark_electrons, read_noise):
    """
    Generates an array representative of pre-digitization electron counts in a detector array with constant signal
    (defined in terms of saturation fraction), a known expected dark current count, and read noise defined by a
    Gaussian standard deviation. Output incorporates signal shot noise.
    """

    electrons = well_depth * saturation_frac * np.ones(image_size)  # expected signal electrons
    electrons = add_dark_electrons(electrons, dark_electrons)  # expected signal plus dark current electrons
    electrons = incorporate_shot_noise(electrons)  # Poisson distribution covering shot noise in signal & dark electrons
    electrons = add_read_noise(electrons, read_noise, well_depth=well_depth)  # Gaussian noise (std=read_noise, mean=0)

    return electrons


def electrons_to_counts(electrons, well_depth, bit_depth=8):

    scaled_values = electrons / well_depth
    max_counts = 2 ** bit_depth
    counts = max_counts * scaled_values
    counts = np.asarray(counts, dtype=np.int64)

    return counts


def rgb_to_pan(image):
    return np.mean(image, axis=2)


def estimate_noise(counts):
    return np.std(counts)


def incorporate_shot_noise(expected_electrons):
    return RNG.poisson(expected_electrons)


def add_read_noise(image_electrons, read_noise, well_depth):
    """
    Incorporates Gaussian read noise, capped at well depth
    """
    read_electrons = RNG.normal(read_noise, size=np.shape(image_electrons))
    image_electrons = image_electrons + read_electrons
    image_electrons = np.clip(image_electrons, 0, well_depth)
    return image_electrons + read_electrons


def add_dark_electrons(image_electrons, dark_count):
    """
    Simple addition of expected dark electrons, with shot noise incorporated after signal and dark electrons
    accumulated
    """
    expected_dark_electrons = dark_count * np.ones_like(image_electrons)  # shot noise added later
    return image_electrons + expected_dark_electrons


def make_image(image_size, saturation_frac, well_depth, dark_electrons, read_noise, bit_depth):
    """
    Returns an image array for constant signal at saturation_frac, converted to an integer, where electrons per count
    is given by well_depth / 2 ** bit_depth
    """
    electrons = pre_sample_electrons(image_size=image_size, saturation_frac=saturation_frac, well_depth=well_depth,
                                     dark_electrons=dark_electrons, read_noise=read_noise)
    counts = electrons_to_counts(electrons, well_depth=well_depth, bit_depth=bit_depth)
    return counts


def estimate_snr(image_size, saturation, well_depth, dark_electrons, read_noise, bit_depth):
    """
    Estimates SNR of an RGB image and it's grayscale counterpoint by comparing a light patch and dark patch, with light
    patch signal specified by saturation and dark patch signal set to zero (before dark current and readout noise).
    Signal is defined here as the mean difference between the light and dark patch, and noise is defined as the standard
    deviation of the difference.
    """

    light_patch = make_image(image_size, saturation_frac=saturation, well_depth=well_depth,
                             dark_electrons=dark_electrons, read_noise=read_noise, bit_depth=bit_depth)
    dark_patch = make_image(image_size, saturation_frac=0, well_depth=well_depth, dark_electrons=dark_electrons,
                            read_noise=read_noise, bit_depth=bit_depth)

    assert np.min(light_patch) >= 0
    assert np.min(dark_patch) >= 0

    diff = light_patch - dark_patch
    noise = estimate_noise(diff)
    snr = np.mean(diff) / noise

    pan_light_path = np.mean(light_patch, axis=2)
    pan_dark_path = np.mean(dark_patch, axis=2)
    pan_diff = pan_light_path - pan_dark_path
    pan_noise = estimate_noise(pan_diff)
    pan_snr = np.mean(pan_diff) / pan_noise

    return snr, noise, pan_snr, pan_noise


def estimate_basic_pan_image_noise(image_size, saturation, well_depth, dark_electrons, read_noise, bit_depth):
    image = make_image(image_size=image_size, saturation_frac=saturation, well_depth=well_depth,
                       dark_electrons=dark_electrons, read_noise=read_noise, bit_depth=bit_depth)
    image = rgb_to_pan(image)
    noise = estimate_noise(image)
    return noise


if __name__ == '__main__':

    _image_size = (64, 64, 3)
    _saturation_frac = 0.5
    _well_depth = 20_000  # electrons
    _dark_electrons = 1000
    _read_noise = 100
    _bit_depth = 8  # note: increasing bit depth increases absolute noise (measured in counts) but increases SNR too

    _rgb_snr, _rgb_noise, _pan_snr, _pan_noise = estimate_snr(image_size=_image_size, saturation=_saturation_frac,
                                                              well_depth=_well_depth, dark_electrons=_dark_electrons,
                                                              read_noise=_read_noise, bit_depth=_bit_depth)

    print(f'SNR (rgb): {round(_rgb_snr, 2)}, {round(_rgb_noise, 2)} counts raw rgb noise  \nSNR (pan):'
          f' {round(_pan_snr, 2)}, {round(_pan_noise, 2)} counts raw pan noise at {_saturation_frac} saturation')
