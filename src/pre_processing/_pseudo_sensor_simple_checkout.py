from src.pre_processing.distortions import coco_tag_to_image_distortions
import argparse
from src.utils.functions import get_config, log_config, id_from_tags
from PIL import Image
import numpy as np
from pathlib import Path
from src.utils.definitions import BASELINE_HIGH_SIGNAL_WELL_DEPTH, \
    PSEUDO_SENSOR_SIGNAL_FRACTIONS, BASELINE_SIGMA_BLUR
from src.pre_processing._pseudo_sensor_varied_snr_checkout import get_output_dir, estimate_snr, strip_plot_2d_array


def main(config):

    level = config['level']
    min_res = config['min_res']
    max_res = config['max_res']
    num_res_fractions = config['num_res_fractions']
    min_blur = BASELINE_SIGMA_BLUR
    num_blur_stds = config['num_blur_stds']
    pseudo_sensor_tag = config['pseudo_sensor_tag']
    name_string = config['name_string']

    output_dir = get_output_dir(name_string=name_string, noise_function_tags=pseudo_sensor_tag)

    config_additions = {'pseudo_sensor_signal_fractions': PSEUDO_SENSOR_SIGNAL_FRACTIONS,
                        'well_depth': BASELINE_HIGH_SIGNAL_WELL_DEPTH,
                        'baseline_sigma_blur': BASELINE_SIGMA_BLUR}
    config.update(config_additions)
    log_config(output_dir=output_dir, config=config)

    digital_number = int((2 ** 8 - 1) * level)
    image = Image.fromarray(digital_number * np.ones((128, 128, 3), dtype=np.uint8))

    res_fractions = np.linspace(min_res, max_res, num=num_res_fractions)
    blur_stds = [min_blur + i for i in range(num_blur_stds)]

    pseudo_sensor = coco_tag_to_image_distortions[pseudo_sensor_tag]

    _data_array_template = np.ones((num_blur_stds, num_res_fractions))
    image_measured_snr_array = -1 * np.ones_like(_data_array_template)
    image_measured_std_array = -1 * np.ones_like(_data_array_template)
    image_mean_array = -1 * np.ones_like(_data_array_template)
    sensor_estimated_snr_array = -1 * np.ones_like(_data_array_template)

    diagnostic_arrays = {}

    for i, blur_std in enumerate(blur_stds):
        for j, res_frac in enumerate(res_fractions):
            sim_image, sensor_est_snr, __, diagnostic_data = pseudo_sensor(image=image,
                                                                           res_frac=res_frac,
                                                                           sigma_blur=blur_std,
                                                                           signal_est_method='mean')

            snr = float(estimate_snr(sim_image))
            std = float(np.std(sim_image))
            mean = float(np.mean(sim_image))
            sensor_estimated_snr = float(sensor_est_snr)

            for key, val in diagnostic_data.items():
                if key not in diagnostic_arrays.keys():
                    diagnostic_arrays[key] = -1 * np.ones_like(_data_array_template)
                    diagnostic_arrays[key][i, j] = val
                else:
                    diagnostic_arrays[key][i, j] = val

            image_measured_snr_array[i, j] = snr
            image_measured_std_array[i, j] = std
            image_mean_array[i, j] = mean
            sensor_estimated_snr_array[i, j] = sensor_estimated_snr

    strip_plot_2d_array(array=image_measured_snr_array,
                        ax0_vals=blur_stds,
                        ax1_vals=res_fractions,
                        ax0_label=r'$\sigma$_blur',
                        ax1_label='resolution',
                        ylabel='SNR',
                        name_stem='image_measured_snr',
                        output_dir=output_dir)

    strip_plot_2d_array(array=sensor_estimated_snr_array,
                        ax0_vals=blur_stds,
                        ax1_vals=res_fractions,
                        ax0_label=r'$\sigma$_blur',
                        ax1_label='resolution',
                        ylabel='SNR',
                        name_stem='sensor_est_snr',
                        output_dir=output_dir)

    strip_plot_2d_array(array=image_mean_array,
                        ax0_vals=blur_stds,
                        ax1_vals=res_fractions,
                        ax0_label=r'$\sigma$_blur',
                        ax1_label='resolution',
                        ylabel='DN',
                        name_stem='image_means',
                        output_dir=output_dir)

    strip_plot_2d_array(array=image_measured_std_array,
                        ax0_vals=blur_stds,
                        ax1_vals=res_fractions,
                        ax0_label=r'$\sigma$_blur',
                        ax1_label='resolution',
                        ylabel='DN',
                        name_stem='image_measured_stds',
                        output_dir=output_dir)

    return diagnostic_arrays


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='config_simple_checkout.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'pseudo_sensor_checkout'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _diagnostics = main(run_config)
