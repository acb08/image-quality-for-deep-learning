import json

import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
from src.d06_chip_study.mtf_study_defitions import baseline_resolution
from src.d06_chip_study.measure_props import get_image_array, load_dataset
from src.d00_utils.definitions import STANDARD_DATASET_FILENAME, REL_PATHS
import shutil
import copy

class PoissonNoiseChannelReplicated(object):
    """
    Generates zero-centered poisson noise with mean selected as described above, divides the noise by 255,
    adds to input tensor and clamps the output to fall on [0, 1]

    Copies the noise across input image channels so that a panchromatic image where the RGB channels are identical
    has the same noise added to each channel.

    """

    def __init__(self, sigma, clamp=True):
        self.clamp = clamp
        self.sigma = sigma

    def __call__(self, tensor):
        channels, height, width = tensor.shape
        noise_shape = (1, height, width)

        lambda_poisson = self.sigma ** 2
        noise = (torch.poisson(torch.ones(noise_shape) * lambda_poisson).float() - lambda_poisson) / 255
        channel_replicated_noise = noise.repeat(channels, 1, 1)
        noisy_tensor = tensor + channel_replicated_noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


# def cr0(size, antialias):
#     transform = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST, antialias=antialias)
#     return transform


def cr1(size, antialias):
    transform = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=antialias)
    return transform


def cr2(size, antialias):
    transform = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=antialias)
    return transform


def cb(std, kernel_size=17):
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def cn(sigma):
    return PoissonNoiseChannelReplicated(sigma)


def distort_chip(chip, distortions):

    transform_list = [transforms.ToTensor()]
    for distortion in distortions:
        transform_list.append(distortion)
    transform = transforms.Compose(transform_list)
    chip = transform(chip)
    chip = transforms.ToPILImage()(chip)

    return chip


def distort_chip_set(dataset, directory, res_transforms, anti_alias_flags, resolutions, blur_stds, noise_stds):

    chip_dir = Path(directory, REL_PATHS['edge_chips'])
    distorted_chip_dir = Path(directory, REL_PATHS['distorted_edge_chips'])

    if distorted_chip_dir.is_dir():
        # Path.rmdir(distorted_chip_dir)
        shutil.rmtree(distorted_chip_dir)
    Path.mkdir(distorted_chip_dir)

    chips_data = dataset['chips']
    distorted_chip_data = {}

    counter = 0

    for interp_method, res_transform in res_transforms:
        for resolution in resolutions:
            for anti_alias_flag in anti_alias_flags:
                for blur_std in blur_stds:
                    for noise_std in noise_stds:

                        distortion_set = []
                        if res_transform:
                            distortion_set.append(res_transform(resolution, anti_alias_flag))
                        if blur_std:
                            distortion_set.append(cb(blur_std))
                        if noise_std:
                            distortion_set.append(cn(noise_std))

                        # distortion_set = [res_transform(resolution, anti_alias_flag), cb(blur_std), cn(noise_std)]

                        for chip_name in chips_data.keys():

                            chip = get_image_array(chip_dir, chip_name)

                            distorted_chip_info = copy.deepcopy(chips_data[chip_name])

                            edge_buffer = distorted_chip_info['edge_buffer']
                            edge_buffer = int((resolution / baseline_resolution) * edge_buffer)
                            distorted_chip_info['edge_buffer'] = edge_buffer

                            parents = distorted_chip_info['parents']
                            parents.append(chip_name)
                            distorted_chip_info['parents'] = parents
                            distortion_info = {
                                'interp_method': interp_method,
                                'res': resolution,
                                'anti_alias_flag': anti_alias_flag,
                                'blur_std': blur_std,
                                'noise_std': noise_std
                            }
                            distorted_chip_info.update(distortion_info)

                            distorted_chip = distort_chip(chip, distortion_set)
                            distorted_chip_name = f"{chip_name.split('.')[0]}_dist_{counter}.png"
                            distorted_chip.save(Path(distorted_chip_dir, distorted_chip_name))

                            distorted_chip_data[distorted_chip_name] = distorted_chip_info

                            counter += 1

    dataset['distorted_chips'] = distorted_chip_data

    with open(Path(directory, STANDARD_DATASET_FILENAME), 'w') as file:
        json.dump(dataset, file)

    print(f'Complete, {counter + 1} distorted chips created')

    return dataset


if __name__ == '__main__':

    # _res_transforms = [('bi-linear', cr1), ('bi-cubic', cr2)]
    # _res_transforms = [('bi-linear', cr1)]
    _res_transforms = [('bi-cubic', cr2)]

    _anti_alias_flags = [True]
    _resolutions = np.arange(int(baseline_resolution / 2), baseline_resolution + 1, step=2)
    _resolutions = [int(res) for res in _resolutions]
    # _resolutions = [32, 48, 64]

    # _blur_stds = np.linspace(0.1, 2, num=10)
    # _blur_stds = [float(std) for std in _blur_stds]
    _blur_stds = [0.1]

    # _noise_stds = np.arange(0, 10, step=2)
    # _noise_stds = [int(std) for std in _noise_stds]
    _noise_stds = [0]

    _directory_key = '0032'

    _directory, _dataset = load_dataset(_directory_key)
    _dataset = distort_chip_set(_dataset, _directory, res_transforms=_res_transforms,
                                anti_alias_flags=_anti_alias_flags, resolutions=_resolutions,
                                blur_stds=_blur_stds, noise_stds=_noise_stds)

    log_file = open(Path(_directory, 'distortion_log_file.txt'), 'w')
    print('resolutions: ', _resolutions, file=log_file)
    # print('resolution transforms: ', , file=log_file)
    print('anti aliasing flags:', _anti_alias_flags, file=log_file)
    print('blur_stds:', _blur_stds, file=log_file)
    print('noise_values:',  _noise_stds, file=log_file)
    log_file.close()
