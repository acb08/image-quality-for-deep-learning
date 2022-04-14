from PIL import Image
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

AIRY_GAUSS_CONVERSION = 1.91


def p2_downsample(image, downsample_step_size):

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


def make_perfect_edge(size, theta, dark_val=0.1, light_val=0.6):

    edge_start = size / 2
    row_indices = np.arange(size)
    edge_locations = edge_start + np.tan(theta / 360) * row_indices
    edge_val = 0.5 * (dark_val + light_val)
    edge_image = light_val * np.ones((size, size))

    for row_idx in row_indices:
        edge_loc = int(edge_locations[row_idx])
        edge_image[row_idx, :edge_loc] = dark_val
        edge_image[row_idx, edge_loc] = edge_val

    return edge_image


def apply_optical_blur(edge_image, kernel_size, sigma):

    edge_image = torch.tensor(edge_image)
    edge_image = torch.unsqueeze(edge_image, dim=0)
    edge_image = transforms.GaussianBlur(kernel_size, sigma=sigma)(edge_image)
    edge_image = torch.squeeze(edge_image, dim=0)
    edge_image = edge_image.numpy()

    return edge_image


def get_blur_parameters(scale_factor, target_q):

    airy_radius = scale_factor * target_q
    std = airy_radius / AIRY_GAUSS_CONVERSION
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


if __name__ == '__main__':

    _image_size = 1024
    _scale_factor = int(1024 / 64)
    _angle = 8
    _q = 1
    # k_size = 15
    # std = 3

    _perfect_edge = make_perfect_edge(_image_size, _angle)
    _std, _kernel_size = get_blur_parameters(_scale_factor, _q)
    _blurred_edge = apply_optical_blur(_perfect_edge, _kernel_size, _std)
    _chip = p2_downsample(_blurred_edge, _scale_factor)

    plot_edges(_perfect_edge, _blurred_edge, [_chip])

