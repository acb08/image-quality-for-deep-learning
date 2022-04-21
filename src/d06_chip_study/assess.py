import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
from src.d06_chip_study.measure_props import get_freq_axis, normed_circ_ap_mtf, load_dataset
from src.d06_chip_study.mtf_study_defitions import parent_names, baseline_resolution
import json


class VectorProps(object):

    def __int__(self, dataset, mtf_lsf):
        self.dataset = dataset
        self.mtf_lsf = mtf_lsf
        self.distorted_chip_data = self.dataset['distorted_edge_chips']
        self.vector_data = self.vectorize_props()

    def vectorize_props(self):

        vector_data = {}
        for i, (chip_id, chip_data) in enumerate(self.distorted_chip_data.items()):

            if i == 0:
                keys = list(chip_data.keys())
                for key in chip_data.keys():
                    vector_data[key] = [chip_data[key]]
            else:
                for key in chip_data.keys():
                    vector_data[key].append(chip_data[key])

        return vector_data


def q2_mtf_compare(mtf, oversample_factor=4, measured_label='measured'):

    f_oversample = get_freq_axis(mtf, oversample_factor=oversample_factor)
    num_freqs_oversampled = len(f_oversample)

    f_cut = f_oversample[int(num_freqs_oversampled / oversample_factor)]
    idx_nyquist = int(num_freqs_oversampled / oversample_factor)
    idx_2x_nyquist = 2 * idx_nyquist
    f = f_oversample[:idx_2x_nyquist]

    q2_mtf = normed_circ_ap_mtf(f, f_cut=f_cut)
    plt.figure()
    plt.plot(f, mtf[:idx_2x_nyquist], label=measured_label)
    plt.plot(f, q2_mtf, label='Q=2 circular aperture')
    plt.xlabel('cycles per pixel')
    plt.ylabel('MTF')
    plt.legend()
    plt.show()


def load_measured_mtf_lsf(directory):
    with open(Path(directory, 'mtf_lsf.json'), 'r') as file:
        data = json.load(file)
    return data


def scale_freq_axis(f, resample_factor):
    return resample_factor * f


def get_nyquist(f_oversampled, oversample_factor):

    num_freqs_oversampled = len(f_oversampled)
    idx_nyquist = int(num_freqs_oversampled / oversample_factor)
    f_nyquist = f_oversampled[idx_nyquist]

    return idx_nyquist, f_nyquist


def truncate(f, mtf, f_cut=0.5):

    f_keep = f[np.where(f <= f_cut)]
    mtf_keep = mtf[np.where(f <= f_cut)]

    return f_keep, mtf_keep


def mtf_evolution_plot(chip_id, dataset, mtf_lsf_data, chip_type_key='chips', directory=None):

    if directory:
        save_dir = Path(directory, REL_PATHS['mtf'])
        save_name = f"{chip_id.split('.')[0]}_mtf_evolution.png"

    chip_mtf = np.asarray(mtf_lsf_data[chip_id]['mtf'])
    oversample_factor = mtf_lsf_data[chip_id]['oversample_factor']

    parent_ids = dataset[chip_type_key][chip_id]['parents']
    parent_mtfs = []
    parent_freq_axes = []
    for parent_id in parent_ids:
        parent_mtf = np.asarray(mtf_lsf_data[parent_id]['mtf'])
        parent_freq_axis = get_freq_axis(parent_mtf)
        parent_freq_axis = scale_freq_axis(parent_freq_axis, len(parent_freq_axis) / len(chip_mtf))
        parent_freq_axis, parent_mtf = truncate(parent_freq_axis, parent_mtf)
        parent_freq_axes.append(parent_freq_axis)
        parent_mtfs.append(parent_mtf)

    freq_axis = get_freq_axis(chip_mtf, oversample_factor=oversample_factor)
    freq_axis, chip_mtf = truncate(freq_axis, chip_mtf)

    plt.figure()
    for i, parent_mtf in enumerate(parent_mtfs):
        plt.plot(parent_freq_axes[i], parent_mtfs[i], label=parent_names[i])
    if chip_type_key == 'chips':
        label = 'chip'
    else:
        label = 'distorted chip'
    plt.plot(freq_axis, chip_mtf, label=label)
    plt.xlabel('cycles per native pixel')
    plt.ylabel('mtf')
    plt.legend()
    if directory:
        plt.savefig(Path(save_dir, save_name))
    plt.show()


def extract_mtf_by_resolution(dataset, mtf_lsf_data, target_interp_method, target_anti_alias_flag, directory=None,
                              legend=True):

    if directory:
        save_dir = Path(directory, REL_PATHS['mtf'], 'res_extract')
        if not save_dir.is_dir():
            Path.mkdir(save_dir)

    distorted_chip_data = dataset['distorted_chips']

    extract = {}

    for chip_id, chip_data in distorted_chip_data.items():

        interp_method = chip_data['interp_method']
        anti_alias_flag = chip_data['anti_alias_flag']

        if interp_method == target_interp_method and anti_alias_flag == target_anti_alias_flag:

            mtf = mtf_lsf_data[chip_id]['mtf']
            rer = mtf_lsf_data[chip_id]['rer']
            snr = mtf_lsf_data[chip_id]['post_distortion_snr']
            res = chip_data['res']
            native_blur = chip_data['native_blur']
            native_noise = chip_data['native_noise']

            if native_blur not in extract.keys():
                extract[native_blur] = {
                    native_noise: {
                        'mtfs': [mtf],
                        'resolutions': [res],
                        'rers': [rer],
                        'snrs': [snr]
                    }
                }
            elif native_noise not in extract[native_blur].keys():
                extract[native_blur][native_noise] = {
                    'mtfs': [mtf],
                    'resolutions': [res],
                    'rers': [rer],
                    'snrs': [snr]
                }
            else:
                extract[native_blur][native_noise]['mtfs'].append(mtf)
                extract[native_blur][native_noise]['resolutions'].append(res)
                extract[native_blur][native_noise]['rers'].append(rer)
                extract[native_blur][native_noise]['snrs'].append(snr)

    for blur_key, sub_dict in extract.items():
        for noise_key, mtf_dict in sub_dict.items():

            plt.figure()
            mtfs = mtf_dict['mtfs']
            rers = mtf_dict['rers']
            snrs = mtf_dict['snrs']
            resolutions = mtf_dict['resolutions']

            for i, mtf in enumerate(mtfs):
                freq_axis = get_freq_axis(mtf)
                resolution = resolutions[i]
                resample_factor = resolution / baseline_resolution
                freq_axis = np.asarray(scale_freq_axis(freq_axis, resample_factor))
                mtf_nyq, freq_axis_nyq = truncate(freq_axis, np.asarray(mtf))
                plt.plot(mtf_nyq, freq_axis_nyq, label=str(resolution))
            plt.title(f"{blur_key} pixel blur, {noise_key} electrons read noise")
            plt.xlabel('cycles per pixel')
            plt.ylabel('MTF')
            if legend:
                plt.legend()
            if directory:
                if legend:
                    save_name = f'res_mtf_leg_{blur_key}_{noise_key}.png'
                else:
                    save_name = f'res_mtf_{blur_key}_{noise_key}.png'
                plt.savefig(Path(save_dir, save_name))
            plt.show()

            plt.figure()
            plt.plot(np.asarray(resolutions) / baseline_resolution, rers)
            plt.xlabel('resolution')
            plt.ylabel('relative edge response')
            plt.title(f"{blur_key} pixel blur, {noise_key} electrons read noise")
            if directory:
                save_name = f'res_rer_{blur_key}_{noise_key}.png'
                plt.savefig(Path(save_dir, save_name))
            plt.show()

            if snrs[0]:
                plt.figure()
                plt.plot(np.asarray(resolutions) / baseline_resolution, snrs)
                plt.xlabel('resolution')
                plt.ylabel('snr')
                plt.title(f"{blur_key} pixel blur, {noise_key} electrons read noise")
                if directory:
                    save_name = f'res_snr_{blur_key}_{noise_key}.png'
                    plt.savefig(Path(save_dir, save_name))
                plt.show()

    return extract


def esf_extract(condition_keys, dataset, mtf_lsf_data, compare_key='res'):

    distorted_chip_data = dataset['distorted_chips']


    pass


if __name__ == '__main__':

    _directory_key = '0028'
    _directory, _dataset = load_dataset(_directory_key)
    _mtf_lsf_data = load_measured_mtf_lsf(_directory)
    _evolution_plots = False
    if _evolution_plots:
        for _chip_id in _dataset['chips'].keys():
            mtf_evolution_plot(_chip_id, _dataset, _mtf_lsf_data, directory=_directory)

        for _chip_id in _dataset['distorted_chips'].keys():
            mtf_evolution_plot(_chip_id, _dataset, _mtf_lsf_data, chip_type_key='distorted_chips', directory=_directory)

    _extract = extract_mtf_by_resolution(_dataset, _mtf_lsf_data, target_interp_method='bi-linear',
                                         target_anti_alias_flag=False, directory=_directory)
    _extract = extract_mtf_by_resolution(_dataset, _mtf_lsf_data, target_interp_method='bi-linear',
                                         target_anti_alias_flag=True, directory=_directory)
    _extract = extract_mtf_by_resolution(_dataset, _mtf_lsf_data, target_interp_method='bi-cubic',
                                         target_anti_alias_flag=False, directory=_directory)
    _extract = extract_mtf_by_resolution(_dataset, _mtf_lsf_data, target_interp_method='bi-cubic',
                                         target_anti_alias_flag=True, directory=_directory)
