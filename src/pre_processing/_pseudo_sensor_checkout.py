from src.pre_processing.distortions import pseudo_sensor_high_noise, pseudo_sensor_med_noise, pseudo_sensor_low_noise
from PIL import Image
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def estimate_snr(img):
    return np.mean(img) / np.std(img)


if __name__ == '__main__':

    image = Image.fromarray(128 * np.ones((128, 128, 3), dtype=np.uint8))
    res_fractions_available = np.linspace(0.25, 1, num=5)

    res_fractions = []

    noise_functions = {'low': pseudo_sensor_low_noise,
                       'med': pseudo_sensor_med_noise,
                       'high': pseudo_sensor_high_noise}

    snrs = {'low': [],
            'med': [],
            'high': []}

    stds = {'low': [],
            'med': [],
            'high': []}

    for i in range(1000):

        res_frac = random.choice(res_fractions_available)
        res_fractions.append(res_frac)

        for key, func in noise_functions.items():
            sim_image, __ = func(image=image, res_frac=res_frac)
            snr = estimate_snr(sim_image)
            std = np.std(sim_image)
            snrs[key].append(snr)
            stds[key].append(std)

    plt.figure()
    for key, snr_list in snrs.items():
        plt.scatter(res_fractions, snr_list, label=key, marker='.', s=1)
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('SNR')
    plt.show()

    plt.figure()
    for key, std_list in stds.items():
        plt.scatter(res_fractions, std_list, label=key, marker='.', s=1)
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('std (DN)')
    plt.show()

