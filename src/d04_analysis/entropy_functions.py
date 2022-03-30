import numpy as np
from scipy.stats import entropy
from scipy import ndimage


def array_entropy(img, base=2):
    """
    :param img: input array (presumed to be an image)
    :param base: base of entropy logarithm, default=2 for entropy in bits
    :return: entropy of input array
    """
    img = np.asarray(img, dtype=np.uint8)
    __, counts = np.unique(img, return_counts=True)

    return entropy(counts, base=base)


def grad_entropy(img):

    img_x, img_y = gradients(img)
    grad_hist = histogram_2d(img_x, img_y)
    gradient_entropy = entropy_2d(grad_hist)

    return 'grad', gradient_entropy


def gradients(img, kernel_id='Sobel'):

    if kernel_id == 'Sobel':
        kernel = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    elif kernel_id == 'Prewitt':
        kernel = np.asarray([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    else:
        return print('Invalid kernel ID')

    img_x = ndimage.convolve(img, kernel, mode='nearest')
    img_y = ndimage.convolve(img, kernel.T, mode='nearest')

    return img_x, img_y


def histogram_2d(img_x, img_y):

    x_grads = np.ravel(img_x)
    y_grads = np.ravel(img_y)
    edges = np.arange(-255, 256)
    grad_hist, __, __ = np.histogram2d(x_grads, y_grads, edges, density=True)

    return grad_hist


def entropy_2d(hist_img):

    prob_hist = hist_img / np.sum(hist_img)
    log_hist = np.copy(prob_hist)
    log_hist[log_hist == 0] = 1
    log_prob = -np.log2(log_hist)
    _entropy = np.sum(prob_hist * log_prob)

    return _entropy


def fourier_magnitude_entropy(img, base=2):
    f_img = image_fft_normalized(img)
    f_img_mag = np.asarray(np.absolute(f_img), dtype=np.uint8)
    return 'fourier_mag', array_entropy(f_img_mag, base=base)


def fourier_2d_entropy(img):
    f_img = image_fft_normalized(img)
    f_img_real = f_img.real
    f_img_imag = f_img.imag
    hist_img = histogram_2d(f_img_real, f_img_imag)
    entropy_f_2d = entropy_2d(hist_img)

    return 'fourier_2d', entropy_f_2d


def image_fft_normalized(img):

    f_img = np.fft.fft2(img, axes=(0, 1))

    n, m = np.shape(img)[:2]
    norm_constant = n * m
    f_img = f_img / norm_constant  # ensures max value <= 255

    return f_img


def shannon_entropy(img):
    return 'shannon', array_entropy(img)


tag_to_entropy_function = {
    'shannon': shannon_entropy,
    'grad': grad_entropy,
    'fourier_mag': fourier_magnitude_entropy,
    'fourier_2d': fourier_2d_entropy
}
