import copy

from src.pre_processing.distortions import coco_tag_to_image_distortions
import argparse
from src.utils.functions import get_config, log_config, id_from_tags
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.definitions import REL_PATHS, ROOT_DIR, BASELINE_HIGH_SIGNAL_WELL_DEPTH, PSEUDO_SENSOR_SIGNAL_FRACTIONS, BASELINE_SIGMA_BLUR


MARKERS = {
    'ps_low_snr': '.',
    'ps_med_low_snr': '+',
    'ps_med_snr_v2': 'v',
    'ps_med_high_snr': 'x',
    'ps_high_snr': '2'
}

LABELS = {
    'ps_low_snr': 'low snr',
    'ps_med_low_snr': 'med-low snr',
    'ps_med_snr_v2': 'med snr',
    'ps_med_high_snr': 'med-high snr',
    'ps_high_snr': 'high snr'
}

SNR_LINE_THRESHOLD = 2


def get_output_dir(name_string, noise_function_tags):

    parent_dir = Path(ROOT_DIR, REL_PATHS['pseudo_sensor_checkout'])
    if not parent_dir.is_dir():
        parent_dir.mkdir()

    if name_string is None:
        tags = noise_function_tags
        if type(tags) == str:
            tags = [tags]
    else:
        tags = [name_string]

    dir_name = id_from_tags(artifact_type='pseudo_sensor_checkout', tags=tags)
    output_dir = Path(parent_dir, dir_name)
    output_dir.mkdir()

    return output_dir


def estimate_snr(img):
    return np.mean(img) / np.std(img)


def remove_latex_eqn(string):
    if '$' not in string:
        return string
    else:
        parts = string.split('$')
        assert len(parts) == 3
        new_string = parts[0] + parts[2]
        return new_string


def strip_plot_2d_array(array, ax0_vals, ax1_vals, ax0_label, ax1_label,
                        ylabel='value', name_stem=None, output_dir=None):

    plt.figure()
    x = ax1_vals
    xlabel = ax1_label
    for i, ax0_val in enumerate(ax0_vals):
        ax0_val = round(ax0_val, 1)
        y = array[i, :]
        plt.plot(x, y, label=f'{ax0_label} = {ax0_val}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.legend()
    if output_dir is not None:
        plt.savefig(Path(output_dir, f'{name_stem}_{remove_latex_eqn(ax1_label)}.png'))
        plt.close()
    else:
        plt.show()

    plt.figure()
    x = ax0_vals
    xlabel = ax0_label
    for i, ax1_val in enumerate(ax1_vals):
        ax1_val = round(ax1_val, 1)
        y = array[:, i]
        plt.plot(x, y, label=f'{ax1_label} = {ax1_val}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.legend()
    if output_dir is not None:
        plt.savefig(Path(output_dir, f'{name_stem}_{remove_latex_eqn(ax0_label)}.png'))
        plt.close()
    else:
        plt.show()


def snr_hist(snrs, output_dir, filename):
    plt.figure()
    plt.hist(snrs)
    plt.savefig(Path(output_dir, filename))


def main(config):

    """
    This function is a spaghetti mess, but not worth cleaning up since it's only used a few times as a sanity check
    """

    level = config['level']
    min_res = config['min_res']
    max_res = config['max_res']
    num_res_fractions = config['num_res_fractions']
    min_blur = BASELINE_SIGMA_BLUR
    num_blur_stds = config['num_blur_stds']
    noise_function_tags = config['noise_function_tags']
    iterations = config['iterations']
    name_string = config['name_string']
    verbose = config['verbose']

    output_dir = get_output_dir(name_string=name_string, noise_function_tags=noise_function_tags)

    config_additions = {'pseudo_sensor_signal_fractions': PSEUDO_SENSOR_SIGNAL_FRACTIONS,
                        'well_depth': BASELINE_HIGH_SIGNAL_WELL_DEPTH,
                        'baseline_sigma_blur': BASELINE_SIGMA_BLUR}
    config.update(config_additions)
    log_config(output_dir=output_dir, config=config)

    digital_number = int((2 ** 8 - 1) * level)
    image = Image.fromarray(digital_number * np.ones((128, 128, 3), dtype=np.uint8))

    res_fractions = np.linspace(min_res, max_res, num=num_res_fractions)
    blur_stds = [min_blur + i for i in range(num_blur_stds)]

    noise_functions = {tag: coco_tag_to_image_distortions[tag] for tag in noise_function_tags}

    _data_array_template = np.ones((num_blur_stds, num_res_fractions))
    image_measured_snr_array_dict = {key: -1 * np.ones_like(_data_array_template) for key in noise_functions}
    image_measured_std_array_dict = {key: -1 * np.ones_like(_data_array_template) for key in noise_functions}
    image_mean_array_dict = {key: -1 * np.ones_like(_data_array_template) for key in noise_functions}
    sensor_estimated_snr_array_dict = {key: -1 * np.ones_like(_data_array_template) for key in noise_functions}

    with open(Path(output_dir, 'log_file.txt'), 'w') as log_file:

        for i, sigma_blur in enumerate(blur_stds):

            sub_dir = Path(output_dir, f'blur-{sigma_blur}')
            sub_dir.mkdir()

            image_measured_snrs = {key: [] for key in noise_functions.keys()}
            stds = {key: [] for key in noise_functions.keys()}
            image_means = {key: [] for key in noise_functions.keys()}
            sensor_estimated_snrs = {key: [] for key in noise_functions.keys()}
            res_fraction_used = []

            all_image_measured_snrs = []
            all_sensor_estimated_snrs = []

            for j in range(iterations):

                deterministic = False
                if j < len(res_fractions):
                    deterministic = True

                if deterministic:
                    res_frac = res_fractions[j]
                else:
                    res_frac = random.choice(res_fractions)

                res_fraction_used.append(res_frac)

                for tag, func in noise_functions.items():
                    sim_image, sensor_estimated_snr, __, diagnostic_data = func(image=image, res_frac=res_frac,
                                                                                sigma_blur=sigma_blur,
                                                                                log_file=log_file
                                                                                )
                    snr = float(estimate_snr(sim_image))
                    std = float(np.std(sim_image))
                    mean = float(np.mean(sim_image))
                    sensor_estimated_snr = float(sensor_estimated_snr['signal_mean_snr'])

                    all_image_measured_snrs.append(snr)
                    all_sensor_estimated_snrs.append(sensor_estimated_snr)

                    image_measured_snrs[tag].append(snr)
                    stds[tag].append(std)
                    image_means[tag].append(mean)
                    sensor_estimated_snrs[tag].append(sensor_estimated_snr)

                    if deterministic:
                        image_measured_snr_array_dict[tag][i, j] = snr
                        image_measured_std_array_dict[tag][i, j] = std
                        image_mean_array_dict[tag][i, j] = mean
                        sensor_estimated_snr_array_dict[tag][i, j] = sensor_estimated_snr

            for tag, data in image_measured_snrs.items():
                print(f'{sigma_blur, tag} measured snr min / max: ', min(data), max(data), file=log_file)
            print('\n', file=log_file)

            for tag, sensor_est_snr_data in sensor_estimated_snrs.items():
                snr_data = image_measured_snrs[tag]
                print(f'{sigma_blur, tag} sensor estimate snr min / max: ', min(sensor_est_snr_data), max(sensor_est_snr_data),
                      file=log_file)
                correlation = np.corrcoef(np.asarray(snr_data), np.asarray(sensor_est_snr_data))[0, 1]
                print(f'estimate / measured snr correlation: ', round(float(correlation), 5), file=log_file)
            print('\n', file=log_file)

            for tag, data in stds.items():
                print(f'{sigma_blur, tag} std min / max: ', min(data), max(data), file=log_file)
            print('\n', file=log_file)

            for tag, data in image_means.items():
                print(f'{sigma_blur, tag} image mean min / max: ', min(data), max(data), file=log_file)
            print('\n', file=log_file)

            min_snr = np.inf
            plt.figure()
            for tag, snr_list in image_measured_snrs.items():
                min_snr = min(min_snr, min(snr_list))
                marker = MARKERS[tag]
                label = LABELS[tag]
                plt.scatter(res_fraction_used, snr_list, label=label, marker=marker, s=1.5)
            if min_snr < SNR_LINE_THRESHOLD:
                plt.plot(res_fraction_used, np.ones(len(res_fraction_used)), ls='--', color='k')
            plt.legend()
            plt.xlabel('resolution fraction')
            plt.ylabel('SNR')
            plt.savefig(Path(sub_dir, f'estimated_snr.png'))
            plt.close()

            for tag, snr_list in image_measured_snrs.items():
                plt.figure()
                marker = MARKERS[tag]
                plt.scatter(res_fraction_used, snr_list, marker=marker, s=2)
                plt.xlabel('resolution fraction')
                if min(snr_list) < SNR_LINE_THRESHOLD:
                    plt.plot(res_fraction_used, np.ones(len(res_fraction_used)), ls='--', color='k')
                plt.ylabel('SNR')
                plt.savefig(Path(sub_dir, f'estimated_snr_{tag}.png'))
                plt.close()

            plt.figure()
            min_snr = np.inf
            for tag, sensor_est_snr_list in sensor_estimated_snrs.items():
                label = LABELS[tag]
                marker = MARKERS[tag]
                min_snr = min(min_snr, min(sensor_est_snr_list))
                plt.scatter(res_fraction_used, sensor_est_snr_list, label=label, marker=marker, s=1.5)
            if min_snr < SNR_LINE_THRESHOLD:
                plt.plot(res_fraction_used, np.ones(len(res_fraction_used)), ls='--', color='k')
            plt.legend()
            plt.xlabel('resolution fraction')
            plt.ylabel('SNR')
            plt.savefig(Path(sub_dir, f'sensor_estimated_snr.png'))
            plt.close()

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.set_aspect('equal', 'box')
            for tag, sensor_est_snr_list in sensor_estimated_snrs.items():
                label = LABELS[tag]
                marker = MARKERS[tag]
                image_measured_snr_list = image_measured_snrs[tag]
                ax.scatter(sensor_est_snr_list, image_measured_snr_list, label=label, marker=marker, s=1.5)
            ax.legend()
            ax.set_xlabel('sensor estimate SNR')
            ax.set_ylabel('image measured SNR')
            plt.savefig(Path(sub_dir, f'sensor_image_snr_compare.png'))
            plt.close()

            for tag, sensor_est_snr_list in sensor_estimated_snrs.items():
                fig, ax = plt.subplots(nrows=1, ncols=1)
                label = LABELS[tag]
                marker = MARKERS[tag]
                ax.set_aspect('equal', 'box')
                image_measured_snr_list = image_measured_snrs[tag]
                ax.scatter(sensor_est_snr_list, image_measured_snr_list, label=label, marker=marker, s=1.5)
                ax.set_xlabel('sensor estimate SNR')
                ax.set_ylabel('image measured SNR')
                plt.savefig(Path(sub_dir, f'{tag}_sensor_image_snr_compare.png'))
                plt.close()

            plt.figure()
            for tag, std_list in stds.items():
                label = LABELS[tag]
                marker = MARKERS[tag]
                plt.scatter(res_fraction_used, std_list, label=label, marker=marker, s=1.5)
            plt.legend()
            plt.xlabel('resolution fraction')
            plt.ylabel('std (DN)')
            plt.savefig(Path(sub_dir, f'std.png'))
            plt.close()

            for tag, std_list in stds.items():
                plt.figure()
                marker = MARKERS[tag]
                plt.scatter(res_fraction_used, std_list, marker=marker, s=2)
                plt.xlabel('resolution fraction')
                plt.ylabel('std (DN)')
                plt.savefig(Path(sub_dir, f'std_{tag}.png'))
                plt.close()

            plt.figure()
            for tag, mean_list in image_means.items():
                label = LABELS[tag]
                marker = MARKERS[tag]
                plt.scatter(res_fraction_used, mean_list, label=label, marker=marker, s=1.5)
            plt.legend()
            plt.xlabel('resolution fraction')
            plt.ylabel('mean (DN)')
            plt.savefig(Path(sub_dir, 'image_means.png'))
            plt.close()

    for key, array in image_measured_snr_array_dict.items():
        strip_plot_2d_array(array=array,
                            ax0_vals=blur_stds,
                            ax1_vals=res_fractions,
                            ax0_label=r'$\sigma$-blur',
                            ax1_label='resolution',
                            ylabel='SNR',
                            name_stem=f'{key}_img_meas_snr',
                            output_dir=output_dir)

    for key, array in image_measured_std_array_dict.items():
        strip_plot_2d_array(array=array,
                            ax0_vals=blur_stds,
                            ax1_vals=res_fractions,
                            ax0_label=r'$\sigma$-blur',
                            ax1_label='resolution',
                            ylabel='std',
                            name_stem=f'{key}_img_meas_std',
                            output_dir=output_dir)

    for key, array in sensor_estimated_snr_array_dict.items():
        strip_plot_2d_array(array=array,
                            ax0_vals=blur_stds,
                            ax1_vals=res_fractions,
                            ax0_label=r'$\sigma$-blur',
                            ax1_label='resolution',
                            ylabel='SNR',
                            name_stem=f'{key}_sensor_est_snr',
                            output_dir=output_dir)

    for key, array in image_mean_array_dict.items():
        strip_plot_2d_array(array=array,
                            ax0_vals=blur_stds,
                            ax1_vals=res_fractions,
                            ax0_label=r'$\sigma$-blur',
                            ax1_label='resolution',
                            ylabel='DN',
                            name_stem=f'{key}_image_mean',
                            output_dir=output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'pseudo_sensor_checkout'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    main(run_config)
