from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import json
from pathlib import Path
from src.d00_utils.functions import key_from_dir
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
import src.d06_chip_study.mtf_study_defitions as DFNS

RNG = np.random.default_rng()


def p2_downsample(image, downsample_step_size=DFNS.scale_factor):

    start_size, compare = np.shape(image)[0], np.shape(image)[1]
    if start_size != compare:
        raise Exception('input image must be square')
    if start_size % downsample_step_size != 0:
        raise Exception('p2_downsample requires downsampling by a power of 2')
    new_size = int(start_size / downsample_step_size)
    chip = np.zeros((new_size, new_size))

    for i in range(new_size):
        v_idx = i * downsample_step_size
        for j in range(new_size):
            h_idx = j * downsample_step_size
            sample = image[v_idx:v_idx + downsample_step_size, h_idx:h_idx + downsample_step_size]
            chip[i, j] = np.mean(sample)

    return chip


def make_perfect_edge(size=DFNS.pre_sample_size, theta=DFNS.angle, dark_val=DFNS.lam_black_reflectance,
                      light_val=DFNS.lam_white_reflectance, half_step=True):

    edge_start = size / 2
    row_indices = np.arange(size)
    edge_locations = edge_start + np.tan(theta / 360) * row_indices
    if half_step:
        edge_val = 0.5 * (dark_val + light_val)
    else:
        edge_val = light_val
    edge_image = light_val * np.ones((size, size), dtype=np.float32)

    for row_idx in row_indices:
        edge_loc = int(edge_locations[row_idx])
        edge_image[row_idx, :edge_loc] = dark_val
        edge_image[row_idx, edge_loc] = edge_val

    return edge_image, edge_locations


def apply_optical_blur(edge_image, kernel_size, sigma):

    edge_image = torch.tensor(edge_image, requires_grad=False)
    edge_image = torch.unsqueeze(edge_image, dim=0)
    edge_image = transforms.GaussianBlur(kernel_size, sigma=sigma)(edge_image)
    edge_image = torch.squeeze(edge_image, dim=0)
    edge_image = edge_image.numpy()

    return edge_image


def get_blur_parameters(target_q, scale_factor=DFNS.scale_factor, conversion=DFNS.airy_gauss_conversion):

    airy_radius = scale_factor * target_q
    std = airy_radius / conversion
    kernel_size = 10 * int(std) + 1

    return std, kernel_size


def plot_edges(perfect_edge, blurred_edge, chips):

    plt.figure()
    plt.imshow(perfect_edge)
    plt.show()

    plt.figure()
    plt.imshow(blurred_edge)
    plt.show()

    for chip in chips:
        plt.figure()
        plt.imshow(chip)
        plt.show()


def convert_to_electrons(image, well_depth=DFNS.well_depth, noise_value=0, dark_electrons=1000):

    image = image * well_depth  # convert from reflectance to radiance
    image = image + dark_electrons  # apply constant dark current across image
    if noise_value:
        image = RNG.poisson(image)  # expected electron counts to Poisson image
        image = RNG.normal(image, noise_value)  # add Gaussian distributed read noise
        if np.max(image) > well_depth:
            print('saturation, ', len(np.where(image > well_depth)[0]), 'pixels')
        if np.min(image) < 0:
            print('zero clipped, ', len(np.where(image < 0)[0]), 'pixels')
        image = np.clip(image, 0, well_depth)

    return image


def electrons_to_dn(image, well_depth=DFNS.well_depth, bit_depth=DFNS.bit_depth):

    conversion = 2**bit_depth / well_depth
    image = conversion * image

    if bit_depth == 8:
        image = np.asarray(image, dtype=np.uint8)
    else:
        image = np.asarray(image, dtype=np.float32)

    return image


def make_edge_chips(q_values, noise_values, half_step, image_size=DFNS.pre_sample_size, theta=DFNS.angle,
                    well_depth=DFNS.well_depth, dark_reflectance=DFNS.lam_black_reflectance,
                    light_reflectance=DFNS.lam_white_reflectance,
                    dark_electrons=DFNS.dark_electrons, scale_factor=DFNS.scale_factor, angle=DFNS.angle):

    perfect_edge, edge_indices = make_perfect_edge(size=image_size, theta=theta, dark_val=dark_reflectance,
                                                   light_val=light_reflectance)

    save_dir_parent = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['mtf_study'])
    key, __, __ = key_from_dir(save_dir_parent)
    save_dir = Path(save_dir_parent, key)
    chip_dir = Path(save_dir, REL_PATHS['edge_chips'])
    edge_dir = Path(save_dir, REL_PATHS['edges'])

    if not chip_dir.is_dir():
        Path.mkdir(chip_dir, parents=True, exist_ok=True)
    if not edge_dir.is_dir():
        Path.mkdir(edge_dir, parents=True, exist_ok=True)

    perfect_edge_save = electrons_to_dn(perfect_edge, well_depth=1)
    perfect_edge_save = Image.fromarray(perfect_edge_save)
    perfect_edge_name = 'perfect_edge.png'
    perfect_edge_save.save(Path(edge_dir, perfect_edge_name))

    stds = []
    kernel_sizes = []
    for q in q_values:
        std, k = get_blur_parameters(q)
        stds.append(std)
        kernel_sizes.append(k)

    kernel_size = max(kernel_sizes)  # just go with the max kernel size

    edge_buffer_left = np.floor(min(edge_indices - (kernel_size + 1)))
    edge_buffer_left = int(np.floor(edge_buffer_left / DFNS.scale_factor)) - 1
    edge_buffer_right = np.ceil(max(edge_indices + (kernel_size + 1)))
    edge_buffer_right = int(np.ceil(edge_buffer_right) / DFNS.scale_factor) + 1
    edge_buffer = min(edge_buffer_left, edge_buffer_right)
    if edge_buffer < 10:
        print(f'Warning: narrow edge buffer: {edge_buffer}')

    chips = {}
    edges = [perfect_edge_name]

    for i, std in enumerate(stds):

        blurred_edge = apply_optical_blur(perfect_edge, kernel_size, std)

        blurred_edge_save = electrons_to_dn(blurred_edge, well_depth=1)
        blurred_edge_save = Image.fromarray(blurred_edge_save)
        blurred_edge_name = f'blurred_edge_{i}.png'
        blurred_edge_save.save(Path(edge_dir, blurred_edge_name))
        edges.append(blurred_edge_name)

        for j, noise_value in enumerate(noise_values):

            # edge = convert_to_electrons(blurred_edge, well_depth, noise_value=noise_value,
            #                             dark_electrons=dark_electrons)
            #
            # edge_save = electrons_to_dn(edge, well_depth=well_depth)
            # edge_save = Image.fromarray(edge_save)
            # blurred_noised_edge_name = f'blurred_noised_edge_{i}_{j}.png'
            # edge_save.save(Path(edge_dir, blurred_noised_edge_name))
            #
            # edges.append(blurred_edge_name)
            # edges.append(blurred_noised_edge_name)
            # chip = p2_downsample(edge, scale_factor)

            chip = p2_downsample(blurred_edge, scale_factor)
            chip = convert_to_electrons(chip, well_depth, noise_value=noise_value, dark_electrons=dark_electrons)
            chip = electrons_to_dn(chip, well_depth)
            snr = measure_snr(chip, edge_buffer)
            chip = Image.fromarray(chip)
            chip_name = f'chip_{i}_{j}.png'
            chips[chip_name] = {
                'native_noise': noise_value,
                'measured_snr': snr,
                'native_blur': std,
                'edge_buffer': edge_buffer,
                'parents': [perfect_edge_name, blurred_edge_name]
            }
            chip.save(Path(chip_dir, chip_name))

    perfect_edge_electrons = convert_to_electrons(perfect_edge, well_depth, noise_value=0,
                                                  dark_electrons=dark_electrons)
    perfect_edge_chip = p2_downsample(perfect_edge_electrons, scale_factor)
    perfect_edge_chip = electrons_to_dn(perfect_edge_chip, well_depth=well_depth)
    perfect_edge_chip = Image.fromarray(perfect_edge_chip)
    perfect_edge_chip_name = 'perfect_edge_chip.png'
    perfect_edge_chip.save(Path(chip_dir, perfect_edge_chip_name))
    chips[perfect_edge_chip_name] = {
        'native_noise': 0,
        'measured_snr': False,
        'native_blur': 0,
        'parents': [perfect_edge_name]
    }

    metadata = {
        'image_size': image_size,
        'scale_factor': scale_factor,
        'angle': angle,
        'dark_offset': dark_electrons,
        'well_depth': well_depth,
        'q_values': q_values,
        'noise_values': noise_values,
        'half_step': half_step
    }

    dataset = {
        'chips': chips,
        'edges': edges,
        'metadata': metadata
    }

    with open(Path(save_dir, STANDARD_DATASET_FILENAME), 'w') as file:
        json.dump(dataset, file)


def measure_snr(chip, buffer):

    dark_region = chip[:, :buffer]
    light_region = chip[:, -buffer:]
    signal = light_region - dark_region
    mean_signal = np.mean(signal)
    noise = np.std(signal)
    if noise > 0:
        snr = mean_signal / noise
    else:
        snr = False

    return snr


if __name__ == '__main__':

    _q_values = [1, 2]
    _noise_values = [0, 100, 1000]
    _half_step = False

    make_edge_chips(_q_values, _noise_values, half_step=_half_step)
