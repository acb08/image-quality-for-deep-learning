import matplotlib.pyplot as plt
import numpy as np
from src.analysis.poisson_sim import normalize, convert_to_uint8, initial_image, image_to_electrons, \
    electrons_to_image
from src.analysis import poisson_sim
from pathlib import Path
from PIL import Image

def check_8_bit_conversions():

    vals = np.arange(256)
    diffs = []
    for val in vals:
        val = np.asarray(val * np.ones((16, 16)), dtype=np.uint8)
        val_scaled = normalize(val)
        val_unscaled = convert_to_uint8(val_scaled)
        diffs.append(np.max(val - val_unscaled))

    print(np.max(diffs), np.min(diffs))


def check_image_conversion(starting_values=None, shape=None):

    if starting_values is None:
        starting_values = np.arange(256)
    if shape is None:
        shape = (64, 64)

    diff_means = []
    diff_stds = []

    for val in starting_values:

        starting_image = initial_image(val / 255, shape=shape, poisson_noise=False)
        electrons = image_to_electrons(starting_image)
        image = electrons_to_image(electrons)

        diff = np.asarray(starting_image) - np.asarray(image)
        diff_means.append(np.mean(diff))
        diff_stds.append(np.std(diff))

    plt.figure()
    plt.plot(starting_values, diff_means)
    plt.title('diff means')
    plt.show()

    plt.figure()
    plt.plot(starting_values, diff_stds)
    plt.title('diff stds')
    plt.show()


if __name__ == '__main__':
    #
    # _starting_values = [1, 8, 16, 32, 64, 128, 200, 255]
    # _shape = (128, 128)
    # check_image_conversion()

    _image = initial_image(poisson_noise=False)
    # _image.show()
    print(np.mean(_image), np.std(_image))
    _electrons = image_to_electrons(_image)
    print(np.mean(_electrons), np.std(_electrons))
    _image_check = electrons_to_image(_electrons)
    print(_image_check == _image)

    check_image_conversion()

    _demo_dir = '/home/acb6595/coco/datasets/train/coco128/images/train2017'
    _image_filenames = list(Path(_demo_dir).iterdir())
    # _image_filenames [filename for filename in _image_filenames]
    _image_filenames = [f for f in _image_filenames if Path(f).is_file()]

    for i in range(3):
        _img = Image.open(_image_filenames[i])
        _img.show()
        _electrons = image_to_electrons(_img)
        _img_check = electrons_to_image(_electrons)
        _img.show()
        _img_check.show()
        _diff = np.asarray(_img) - np.asarray(_img_check)
        print(np.mean(_diff), np.std(_diff))


    # #
    # _starting_val = 128
    # _starting_image = initial_image(_starting_val, shape=_shape)

    # _dc_fractions = np.linspace(0, 0.9, num=10, endpoint=True)
    # _blur_stds = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

    # _dc_fractions = 10
    #
    # _snr_vals = []
    # _mean_signal_vals = []
    #
    # for _starting_val in _starting_values:
    #
    #     _image = initial_image(_starting_val, _shape)
    #     _electrons = image_to_electrons(_image)
    #     _poisson_electrons = apply_partial_poisson_distribution(_electrons, dc_fraction=0)
    #     _mean_signal = np.mean(_electrons)
    #     _noise = np.std(_poisson_electrons)
    #     _snr = _mean_signal / _noise
    #
    #     _snr_vals.append(_snr)
    #     _mean_signal_vals.append(_mean_signal)
    #
    # _ideal_snrs = [np.sqrt(mean_electrons) for mean_electrons in _mean_signal_vals]
    #
    # plt.figure()
    # plt.plot(_mean_signal_vals, _snr_vals)
    # plt.plot(_mean_signal_vals, _ideal_snrs)
    # plt.show()
    #

    # _starting_value = 128
    #
    # _noise_iterations = 10
    #
    # _dc_fractions = np.linspace(0, 0.9, num=10, endpoint=True)
    #
    # _dc_frac_stds = {}
    #
    # for _dc_frac in _dc_fractions:
    #
    #     _stds = []
    #     _signal = initial_signal(_starting_value, shape=_shape)
    #
    #     for i in range(_noise_iterations):
    #         _signal = apply_partial_poisson_distribution(_signal, _dc_frac)
    #         _std = np.std(_signal)
    #         _stds.append(_std)
    #
    #     _dc_frac_stds[_dc_frac] = _stds
    #
    # plt.figure()
    # for _dc_frac, _stds in _dc_frac_stds.items():
    #     plt.plot(_stds, label=round(_dc_frac, 1))
    # plt.legend()
    # plt.show()
