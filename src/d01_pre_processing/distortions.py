from PIL import Image
import torch
import numpy as np
import os
import json
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import random


class AddPoissonNoise(object):

    """
    Adds zero-centered Poisson noise of standard deviation sigma_poisson.

    sigma_poisson is intended to be specified in digital number to be added to an 8-bit image, but it is applied
    assuming the image has already been scaled to fall on [0, 1], so the zero-centered Poisson distribution
    is divided by 255 to match the input image scaling.
    """

    def __init__(self, sigma_poisson, clamp=True):
        self.lambda_poisson = sigma_poisson ** 2
        self.clamp = clamp

    def __call__(self, tensor):
        noise = (torch.poisson(torch.ones(tensor.shape) * self.lambda_poisson).float() - self.lambda_poisson) / 255
        noisy_tensor = tensor + noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class AddVariablePoissonNoise(object):
    """
    Selects Poisson noise standard deviation on sigma_poisson_range with uniform probability
    inclusive of the endpoints.

    generates zero-centered poisson noise with mean selected as described above, divides the noise by 255,
    adds to input tensor and clamps the output to fall on [0, 1]



    """
    def __init__(self, sigma_poisson_range, clamp=True):
        self.low = sigma_poisson_range[0] ** 2
        self.high = sigma_poisson_range[1] ** 2 + 1  # add one because torch.randint excludes right endpoint
        self.clamp = clamp

    def __call__(self, tensor):
        lambda_poisson = torch.randint(self.low, self.high, (1,))
        noise = (torch.poisson(torch.ones(tensor.shape) * lambda_poisson).float() - lambda_poisson) / 255
        noisy_tensor = tensor + noise
        if self.clamp:
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class VariableResolution(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.transform_bank = self.build_transform_bank()

    def build_transform_bank(self):
        transform_bank = []
        for size in self.sizes:
            size = int(size)
            new_transform = transforms.Resize(size,
                                              interpolation=transforms.InterpolationMode.BILINEAR,
                                              antialias=True)
            transform_bank.append(new_transform)
        print('build_transform_bank called')
        return transform_bank

    def __call__(self, tensor):
        transform_use = random.choice(self.transform_bank)
        return transform_use(tensor)


def pan(img):
    """
    :param img: torch image tensor
    :param shape: keword arg to ignore, enables compatibility with other distortion functions
    :return: torch image tensor, converted to grayscale
    """

    return transforms.Grayscale(num_output_channels=3)(img), 'mode', 'pan'


def pan_c():
    return transforms.Grayscale(num_output_channels=3)


def b3(img):

    kernel_size = 15
    sigma_range = np.linspace(0.1, 3.5, num=50, endpoint=True)
    std = np.random.choice(sigma_range)

    if not isinstance(img, torch.Tensor):
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std), 'std', std
    else:
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def bc3(img):

    """
    Coarser version of b3, with only 20 stds rather than 52.
    """

    kernel_size = 15
    sigma_range = np.linspace(0.1, 3.5, num=21, endpoint=True)
    std = np.random.choice(sigma_range)

    if not isinstance(img, torch.Tensor):
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std), 'std', std
    else:
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b0(img):
    """
    a very simple function that doesn't apply any blur:)
    """
    if not isinstance(img, torch.Tensor):
        return None, 'mode', 'pan'
    else:
        return img, 'std', 0


def b4():
    """
    constant blur transform at the midpoint of the b3 distortion space
    """
    kernel_size = 15
    std = 1.75
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def b5():
    """
    constant blur transform at the endpoint of the b3 distortion space
    """
    kernel_size = 15
    std = 3.5

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def b6(img):

    kernel_size = 15
    std = 1.75

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b7(img):

    kernel_size = 15
    std = 1.75

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b8(img):

    kernel_size = 15
    std = 3.5

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def b9():
    """
    variable blur transform centered around the midpoint of the b3 distortion space
    """
    kernel_size = 15
    std = (1.5, 2)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def b10():
    """
    variable blur transform to be used in place of b3, with blurring accomplished at batch level
    """
    kernel_size = 15
    std = (0.1, 3.5)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def br1():
    """
    variable blur transform covering the first half of the 0 - 3.5 pixel blur space
    """
    kernel_size = 15
    std = (0.1, 1.75)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def br2():
    """
    variable blur transform covering the second half of the 0 - 3.5 pixel blur space
    """
    kernel_size = 15
    std = (1.75, 3.5)

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)


def bp1(img):
    """
    Center blur point (hence the "p") for lower blur range in distortion octant scheme.
    """

    kernel_size = 15
    std = 0.875  # i.e. 0.25 * 3.5

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def bp2(img):
    """
    Center blur point (hence the "p") for upper blur range in distortion octant scheme.
    """

    kernel_size = 15
    std = 2.625  # i.e. 0.75 * 3.5

    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def n2(img):

    """

    Stop Using this version. This noise version does not actually add noise, instead returning
    the original image.

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    """

    shape = img.shape
    lambda_poisson = np.random.randint(0, 20)
    noise = torch.poisson(torch.ones(shape) * lambda_poisson).float() / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img, 'lambda_poisson', lambda_poisson


def n3(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    """

    shape = img.shape
    lambda_poisson = np.random.randint(0, 20)
    noise = torch.poisson(torch.ones(shape) * lambda_poisson).float() / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def nf3(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    Fixed version of n3, for us in generating testing images.

    """

    shape = img.shape
    sigma_poisson = np.random.randint(0, 21)  # add 1 to target distribution, high is "one above the highest integer to
    # be drawn from the target distribution
    lambda_poisson = sigma_poisson**2
    noise = (torch.poisson(torch.ones(shape) * lambda_poisson).float() - lambda_poisson) / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def n4():
    """
    returns a custom transform that adds mean of 10 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose().
    """
    lambda_poisson = 10
    clamp = True
    return AddPoissonNoise(lambda_poisson, clamp)


def nf4():
    """
    returns a custom transform that adds mean of 10 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose().
    """
    lambda_poisson = 10
    clamp = True
    return AddPoissonNoise(lambda_poisson, clamp)


def n5():
    """
    returns a custom transform that adds mean of 20 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose(). Designed as the endpoint of the n3 distortion space
    """
    lambda_poisson = 20
    clamp = True
    return AddPoissonNoise(lambda_poisson, clamp)


def nf5():
    """
    returns a custom transform that adds mean of 20 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose(). Designed as the endpoint of the n3 distortion space
    """
    lambda_poisson = 20
    clamp = True
    return AddPoissonNoise(lambda_poisson, clamp)


def n6(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    """

    shape = img.shape
    lambda_poisson = 5
    noise = torch.poisson(torch.ones(shape) * lambda_poisson).float() / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def n7(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    """

    shape = img.shape
    lambda_poisson = 15
    noise = torch.poisson(torch.ones(shape) * lambda_poisson).float() / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def n8(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    """

    shape = img.shape
    lambda_poisson = 10
    noise = torch.poisson(torch.ones(shape) * lambda_poisson).float() / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def n9(img):

    """
    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.
    """

    shape = img.shape
    lambda_poisson = 20
    noise = torch.poisson(torch.ones(shape) * lambda_poisson).float() / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def n10():
    """
    returns a custom transform that adds 5 - 15 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose(). Designed as a mid-band training image transform for a 20 DN Poisson noise
    space
    """
    lambda_poisson_range = (5, 15)
    clamp = True
    return AddVariablePoissonNoise(lambda_poisson_range, clamp)


def nf10():
    """
    returns a custom transform that adds 5 - 15 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose(). Designed as a mid-band training image transform for a 20 DN Poisson noise
    space

    n10, but zero-centered noise added, intended for use after fixing Poisson noise class to fix lambda/sigma mix up
    """
    lambda_poisson_range = (5, 15)
    clamp = True
    return AddVariablePoissonNoise(lambda_poisson_range, clamp)


def nf11():
    """
    returns a custom transform that adds 0 - 20 DN Poisson noise scaled by 1 /255 and clamps the final
    output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson
    to be used in Transforms.compose().


    Replacement for n3 in model training, where images are distorted at the batch level

    """
    lambda_poisson_range = (0, 20)
    clamp = True
    return AddVariablePoissonNoise(lambda_poisson_range, clamp)


def nfr1():
    """
    returns a custom transform that adds 0 - 10 DN Poisson noise that is zero-centered and scaled by 1 /255, with
    clamped the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson. Intended
    to be used in Transforms.compose().
    """
    lambda_poisson_range = (0, 10)
    clamp = True
    return AddVariablePoissonNoise(lambda_poisson_range, clamp)


def nfr2():
    """
    returns a custom transform that adds 10 - 20 DN Poisson noise that is zero-centered and scaled by 1 /255, with
    clamped the final output tensor to fall on [0, 1], where output_tensor = input_tensor + Poisson. Intended
    to be used in Transforms.compose().
    """
    lambda_poisson_range = (10, 20)
    clamp = True
    return AddVariablePoissonNoise(lambda_poisson_range, clamp)


def nfp1(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    Center noise point (hence the "p") for lower noise range in distortion octant scheme.

    """

    shape = img.shape
    sigma_poisson = 5
    # be drawn from the target distribution
    lambda_poisson = sigma_poisson**2
    noise = (torch.poisson(torch.ones(shape) * lambda_poisson).float() - lambda_poisson) / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def nfp2(img):

    """

    :param img: image tensor, values on range [0,1]
    :return: image + Poisson noise, where poisson noise is scaled by 1 / 255  and
    resulting image is clamped to fall on [0, 1]

    Note: if img is grayscale but 3-channel, effective noise will be reduced
    relative to adding noise to a 1-channel image by virtue of averaging across
    channels for a 3-channel image.

    Center noise point (hence the "p") for upper noise range in distortion octant scheme.

    """

    shape = img.shape
    sigma_poisson = 15
    # be drawn from the target distribution
    lambda_poisson = sigma_poisson**2
    noise = (torch.poisson(torch.ones(shape) * lambda_poisson).float() - lambda_poisson) / 255
    img_out = torch.clone(img) + noise
    img_out = torch.clamp(img_out, 0, 1)

    return img_out, 'lambda_poisson', lambda_poisson


def r1(img):
    """
    a very simple function that doesn't change image resolution :)
    """
    return img, 'res', 1.0


def r2(img):

    """
    scales image down by a factor on [0.5, 1] and re-sizes to original size, emulating a lower resolution
    image

    :param img:
    :return:
    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale_range = np.linspace(0.5, 1, num=52, endpoint=True)
    scale = np.random.choice(scale_range, replace=True)
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Resize(start_min_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def r3(img):

    """
    scales image down by a factor on [0.5, 1] WITHOUT resizing to original size, emulating a lower resolution
    image.

    :param img:
    :return:
    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale_range = np.linspace(0.5, 1, num=52, endpoint=True)
    scale = np.random.choice(scale_range, replace=True)
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def rc3(img):

    """
    scales image down by a factor on [0.5, 1] WITHOUT resizing to original size, emulating a lower resolution
    image. A coarser version of r3, with only 21 resolution bins rather than 52

    :param img:
    :return:
    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale_range = np.linspace(0.5, 1, num=21, endpoint=True)
    scale = np.random.choice(scale_range, replace=True)
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def r4():

    """
    returns transform to down-scale a 256 x 256 image to 75% scale (i.e. 192 x 192)
    """
    downsize_dim = 192
    return transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)


def r5(img):

    """
    scales image down by a factor on [0.3, 1] WITHOUT resizing to original size, emulating a lower resolution
    image.

    :param img:
    :return:
    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale_range = np.linspace(0.3, 1, num=52, endpoint=True)
    scale = np.random.choice(scale_range, replace=True)
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def r6():

    """
    returns transform to down-scale a 256 x 256 image to 50% scale (i.e. 128 x 128). Intended as the endpoint of the
    r3 distortion space.
    """
    downsize_dim = 128
    return transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)


def r7(img):

    """
    scales image down by 0.75 WITHOUT resizing to original size, emulating a lower resolution
    image. Equivalent to r4, but for use in scripts where each distortion is called separately rather than in
    transforms.Compose()
    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale = 0.75
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def r8(img):

    """
    scales image down by 0.5 WITHOUT resizing to original size, emulating a lower resolution
    image. Equivalent to r6, but for use in scripts where each distortion is called separately rather than in
    transforms.Compose()
    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale = 0.5
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def r9():
    """
    returns transform to down-scale a 256 x 256 image to ~65% - 85% scale (i.e. 166 x 166 to 218 x 218).
    Mid-band variable resolution downsampling function centered at the midpoint of the r3 transform.
    """

    min_size = 166
    max_size = 218
    downsize_dims = np.arange(min_size, max_size + 1)
    return VariableResolution(downsize_dims)


def r10():
    """
    returns transform to down-scale a 256 x 256 image to 50% - 100% scale (i.e. 128 x 128 to 256 x 256).

    Intended as replacement for r3, with resizing now done at the batch level.

    """

    min_size = 128
    max_size = 256
    downsize_dims = np.arange(min_size, max_size + 1)
    return VariableResolution(downsize_dims)


def rr1():
    """
    returns transform to down-scale a 256 x 256 image to 75% - 100% scale (i.e. 192 x 192 to 256 x 256).
    """

    min_size = 192
    max_size = 256
    downsize_dims = np.arange(min_size, max_size + 1)

    return VariableResolution(downsize_dims)


def rr2():
    """
    returns transform to down-scale a 256 x 256 image to 50% - 75% scale (i.e. 128 x 128 to 192 x 192).
    """

    min_size = 128
    max_size = 192
    downsize_dims = np.arange(min_size, max_size + 1)

    return VariableResolution(downsize_dims)


def rp1(img):

    """
    scales image down by 12.25% WITHOUT resizing to original size, emulating a lower resolution
    image.

    Center resolution point (hence the "p") for higher resolution range in distortion octant scheme.

    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale = 0.875
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def rp2(img):
    """
    scales image down by 12.25% WITHOUT resizing to original size, emulating a lower resolution
    image.

    Center resolution point (hence the "p") for lower resolution range in distortion octant scheme.

    """

    shape = img.shape[-2:]
    start_min_dim = min(shape)

    scale = 0.675
    downsize_dim = int(scale * start_min_dim)

    res_transform = transforms.Compose([
        transforms.Resize(downsize_dim, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    ])

    return res_transform(img), 'res', scale


def tag_to_func(tag):  # TODO: see if calling an intermediate function is slower than import function directly

    """

    Repeatably maps distortion tags to stable distortion functions. Allows distortion scripts to
    import a single function that then calls the appropriate distortion functions. The more I think about it,
    I should probably just import the distortion functions themselves...

    :param tag: tag mapped to a particular distortion function
    :return: distortion function corresponding to tag
    """

    if tag == 'pan':
        return pan
    if tag == 'pan_c':
        return pan_c()
    if tag == 'pan_br':  # pan branch (hence '_br') to avoid overwriting other pan test datasets
        return pan       # this is why I need a new scheme for config mgt/saving data...
    if tag == 'pan_crop':
        return pan
    if tag == 'pan_msc1':  # msc stands for mega set coarse
        return pan
    if tag == 'pan_msc2':
        return pan
    # tags to avoid overwriting parent images in octant scheme
    if tag == 'pan_p111':
        return pan
    if tag == 'pan_p112':
        return pan
    if tag == 'pan_p121':
        return pan
    if tag == 'pan_p122':
        return pan
    if tag == 'pan_p211':
        return pan
    if tag == 'pan_p212':
        return pan
    if tag == 'pan_p221':
        return pan
    if tag == 'pan_p222':
        return pan

    if tag == 'b0':
        return b0
    if tag == 'b3':
        return b3
    if tag == 'bc3':
        return bc3
    if tag == 'b4':
        return b4()
    if tag == 'b5':
        return b5()
    if tag == 'b6':
        return b6
    if tag == 'b7':
        return b7
    if tag == 'b8':
        return b8
    if tag == 'b9':
        return b9()
    if tag == 'b10':
        return b10()
    if tag == 'br1':
        return br1()
    if tag == 'br2':
        return br2()
    if tag == 'bp1':
        return bp1
    if tag == 'bp2':
        return bp2

    ######################################
    # n2 - n10 all incorrectly incorporate Poisson noise.
    # if tag == 'n2':
    #     return n2
    # if tag == 'n3':
    #     return n3
    # if tag == 'n4':
    #     return n4()
    # if tag == 'n5':
    #     return n5()
    # if tag == 'n6':
    #     return n6
    # if tag == 'n7':
    #     return n7
    # if tag == 'n8':
    #     return n8
    # if tag == 'n9':
    #     return n9
    # if tag == 'n10':
    #     return n10()
    #################################
    # corrected noise tags below
    if tag == 'nf3':
        return nf3
    if tag == 'nf4':
        return nf4()
    if tag == 'nf5':
        return nf5()
    if tag == 'nf10':
        return nf10()
    if tag == 'nf11':
        return nf11()
    # "octant transforms"
    if tag == 'nfr1':
        return nfr1()
    if tag == 'nfr2':
        return nfr2()
    if tag == 'nfp1':
        return nfp1
    if tag == 'nfp2':
        return nfp2

    if tag == 'r1':
        return r1
    if tag == 'r2':
        return r2
    if tag == 'r3':
        return r3
    if tag == 'rc3':
        return rc3
    if tag == 'r3_br': # another overwrite project branch for making mega dataset
        return r3
    if tag == 'r3_br2':
        return r3
    if tag == 'r4':
        return r4()
    if tag == 'r5':
        return r5
    if tag == 'r6':
        return r6()
    if tag == 'r7':
        return r7
    if tag == 'r8':
        return r8
    if tag == 'r9':
        return r9()
    if tag == 'r10':
        return r10()
    if tag == 'rr1':
        return rr1()
    if tag == 'rr2':
        return rr2()
    if tag == 'rp1':
        return rp1
    if tag == 'rp2':
        return rp2


def test_noise_transform(noise_func):

    t = transforms.Compose(noise_func)
    runs_per_tensor = 10

    input_tensors = [
        torch.rand((3, 15, 15)),
        torch.zeros((3, 15, 15)),
        torch.ones((1, 15, 15))
    ]

    mean_diffs = []

    for input_tensor in input_tensors:

        run_mean_diffs = []

        for run in range(runs_per_tensor):
            output = t(input_tensor)
            _diff = output - input_tensor
            mean_diff = torch.mean(255 * _diff)
            run_mean_diffs.append(float(mean_diff))

        mean_diffs.append(run_mean_diffs)

    return mean_diffs


def test_res_transform(res_func):

    input_tensor = torch.rand((10, 3, 256, 256))
    num_iterations = 100
    t = transforms.Compose(res_func)
    sizes = []
    for i in range(num_iterations):
        output_tensor = t(input_tensor)
        sizes.append(output_tensor.shape)

    return sizes


def test_blur_function(blur_func):

    input_tensor = torch.rand((10, 3, 256, 256))
    num_iterations = 100
    t = transforms.Compose(blur_func)
    outputs = []
    for i in range(num_iterations):
        output_tensor = t(input_tensor)
        outputs.append(output_tensor)
    return outputs

