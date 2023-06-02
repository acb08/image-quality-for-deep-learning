from src.pre_processing.distortions import coco_tag_to_image_distortions
import argparse
from src.utils.functions import get_config, log_config, id_from_tags
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.definitions import REL_PATHS, ROOT_DIR, WELL_DEPTH, PSEUDO_SENSOR_SIGNAL_FRACTIONS


MARKERS = {
    'ps_low_snr': '.',
    'ps_med_snr': 'v',
    'ps_high_snr': '2'
}

LABELS = {
    'ps_low_snr': 'low snr',
    'ps_med_snr': 'mid snr',
    'ps_high_snr': 'high snr'
}

SNR_LINE_THRESHOLD = 2


def get_output_dir(name_string, noise_function_tags):

    parent_dir = Path(ROOT_DIR, REL_PATHS['pseudo_sensor_checkout'])
    if not parent_dir.is_dir():
        parent_dir.mkdir()

    if name_string is None:
        tags = noise_function_tags
    else:
        tags = [name_string]

    dir_name = id_from_tags(artifact_type='pseudo_sensor_checkout', tags=tags)
    output_dir = Path(parent_dir, dir_name)
    output_dir.mkdir()

    return output_dir


def estimate_snr(img):
    return np.mean(img) / np.std(img)


def main(config):

    level = config['level']
    min_res = config['min_res']
    max_res = config['max_res']
    num_res_fractions = config['num_res_fractions']
    noise_function_tags = config['noise_function_tags']
    iterations = config['iterations']
    name_string = config['name_string']
    verbose = config['verbose']

    output_dir = get_output_dir(name_string=name_string, noise_function_tags=noise_function_tags)

    config_additions = {'pseudo_sensor_signal_fractions': PSEUDO_SENSOR_SIGNAL_FRACTIONS,
                        'well_depth': WELL_DEPTH}
    config.update(config_additions)

    log_config(output_dir=output_dir, config=config)

    digital_number = int((2 ** 8 - 1) * level)
    image = Image.fromarray(digital_number * np.ones((128, 128, 3), dtype=np.uint8))

    res_fractions = np.linspace(min_res, max_res, num=num_res_fractions)

    noise_functions = {tag: coco_tag_to_image_distortions[tag] for tag in noise_function_tags}

    image_measured_snrs = {key: [] for key in noise_functions.keys()}
    stds = {key: [] for key in noise_functions.keys()}
    image_means = {key: [] for key in noise_functions.keys()}
    sensor_estimated_snrs = {key: [] for key in noise_functions.keys()}

    res_fraction_used = []

    with open(Path(output_dir, 'log_file.txt'), 'w') as log_file:

        for i in range(iterations):

            res_frac = random.choice(res_fractions)
            res_fraction_used.append(res_frac)

            for key, func in noise_functions.items():
                sim_image, sensor_estimated_snr, __, approx_noise_dn = func(image=image, res_frac=res_frac,
                                                                            verbose=verbose,
                                                                            log_file=log_file,
                                                                            signal_est_method='mean')
                snr = float(estimate_snr(sim_image))
                std = float(np.std(sim_image))
                mean = float(np.mean(sim_image))
                image_measured_snrs[key].append(snr)
                stds[key].append(std)
                image_means[key].append(mean)
                sensor_estimated_snrs[key].append(float(sensor_estimated_snr))

        for key, data in image_measured_snrs.items():
            print(f'{key} measured snr min / max: ', min(data), max(data), file=log_file)
        print('\n', file=log_file)

        for key, sensor_est_snr_data in sensor_estimated_snrs.items():
            snr_data = image_measured_snrs[key]
            print(f'{key} sensor estimate snr min / max: ', min(sensor_est_snr_data), max(sensor_est_snr_data),
                  file=log_file)
            correlation = np.corrcoef(np.asarray(snr_data), np.asarray(sensor_est_snr_data))[0, 1]
            print(f'estimate / measured snr correlation: ', round(float(correlation), 5), file=log_file)
        print('\n', file=log_file)

        for key, data in stds.items():
            print(f'{key} std min / max: ', min(data), max(data), file=log_file)
        print('\n', file=log_file)

        for key, data in image_means.items():
            print(f'{key} image mean min / max: ', min(data), max(data), file=log_file)
        print('\n', file=log_file)

    min_snr = np.inf
    plt.figure()
    for key, snr_list in image_measured_snrs.items():
        min_snr = min(min_snr, min(snr_list))
        marker = MARKERS[key]
        label = LABELS[key]
        plt.scatter(res_fraction_used, snr_list, label=label, marker=marker, s=1.5)
    if min_snr < SNR_LINE_THRESHOLD:
        plt.plot(res_fraction_used, np.ones(len(res_fraction_used)), ls='--', color='k')
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('SNR')
    plt.savefig(Path(output_dir, 'estimated_snr.png'))

    for key, snr_list in image_measured_snrs.items():
        plt.figure()
        marker = MARKERS[key]
        plt.scatter(res_fraction_used, snr_list, marker=marker, s=2)
        plt.xlabel('resolution fraction')
        if min(snr_list) < SNR_LINE_THRESHOLD:
            plt.plot(res_fraction_used, np.ones(len(res_fraction_used)), ls='--', color='k')
        plt.ylabel('SNR')
        plt.savefig(Path(output_dir, f'estimated_snr_{key}.png'))

    plt.figure()
    min_snr = np.inf
    for key, sensor_est_snr_list in sensor_estimated_snrs.items():
        label = LABELS[key]
        marker = MARKERS[key]
        min_snr = min(min_snr, min(sensor_est_snr_list))
        plt.scatter(res_fraction_used, sensor_est_snr_list, label=label, marker=marker, s=1.5)
    if min_snr < SNR_LINE_THRESHOLD:
        plt.plot(res_fraction_used, np.ones(len(res_fraction_used)), ls='--', color='k')
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('SNR')
    plt.savefig(Path(output_dir, 'sensor_estimated_snr.png'))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_aspect('equal', 'box')
    for key, sensor_est_snr_list in sensor_estimated_snrs.items():
        label = LABELS[key]
        marker = MARKERS[key]
        image_measured_snr_list = image_measured_snrs[key]
        ax.scatter(sensor_est_snr_list, image_measured_snr_list, label=label, marker=marker, s=1.5)
    ax.legend()
    ax.set_xlabel('sensor estimate SNR')
    ax.set_ylabel('image measured SNR')
    plt.savefig(Path(output_dir, 'sensor_image_snr_compare.png'))

    for key, sensor_est_snr_list in sensor_estimated_snrs.items():
        fig, ax = plt.subplots(nrows=1, ncols=1)
        label = LABELS[key]
        marker = MARKERS[key]
        ax.set_aspect('equal', 'box')
        image_measured_snr_list = image_measured_snrs[key]
        ax.scatter(sensor_est_snr_list, image_measured_snr_list, label=label, marker=marker, s=1.5)
        ax.set_xlabel('sensor estimate SNR')
        ax.set_ylabel('image measured SNR')
        plt.savefig(Path(output_dir, f'{key}_sensor_image_snr_compare.png'))

    plt.figure()
    for key, std_list in stds.items():
        label = LABELS[key]
        marker = MARKERS[key]
        plt.scatter(res_fraction_used, std_list, label=label, marker=marker, s=1.5)
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('std (DN)')
    plt.savefig(Path(output_dir, 'std.png'))

    for key, std_list in stds.items():
        plt.figure()
        marker = MARKERS[key]
        plt.scatter(res_fraction_used, std_list, marker=marker, s=2)
        plt.xlabel('resolution fraction')
        plt.ylabel('std (DN)')
        plt.savefig(Path(output_dir, f'std_{key}.png'))

    plt.figure()
    for key, mean_list in image_means.items():
        label = LABELS[key]
        marker = MARKERS[key]
        plt.scatter(res_fraction_used, mean_list, label=label, marker=marker, s=1.5)
    plt.legend()
    plt.xlabel('resolution fraction')
    plt.ylabel('mean (DN)')
    plt.savefig(Path(output_dir, 'image_means.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'pseudo_sensor_checkout'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    main(run_config)
