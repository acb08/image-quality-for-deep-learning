from src.pre_processing.distortions import pseudo_sensor_low_snr, pseudo_sensor_med_snr, pseudo_sensor_high_snr
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

    noise_functions = {'low_snr': pseudo_sensor_low_snr,
                       'med_snr': pseudo_sensor_med_snr,
                       'high_snr': pseudo_sensor_high_snr}

    snrs = {'low_snr': [],
            'med_snr': [],
            'high_snr': []}

    stds = {'low_snr': [],
            'med_snr': [],
            'high_snr': []}

    image_means = {'low_snr': [],
                   'med_snr': [],
                   'high_snr': []}

    sensor_estimated_snrs = {'low_snr': [],
                             'med_snr': [],
                             'high_snr': []}

    for i in range(100):

        res_frac = random.choice(res_fractions_available)
        res_fractions.append(res_frac)

        for key, func in noise_functions.items():
            sim_image, sensor_estimated_snr, __, __ = func(image=image, res_frac=res_frac)
            snr = estimate_snr(sim_image)
            std = np.std(sim_image)
            mean = np.mean(sim_image)
            snrs[key].append(snr)
            stds[key].append(std)
            image_means[key].append(mean)
            sensor_estimated_snrs[key].append(sensor_estimated_snr)

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

    plt.figure()
    for key, mean_list in image_means.items():
        plt.scatter(res_fractions, mean_list, label=key, marker='.', s=1)
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('mean (DN)')
    plt.show()

