import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
from src.d05_rer.measure import get_freq_axis, normed_circ_ap_mtf, load_dataset
from src.d05_rer.rer_defs import parent_names
import json


class VectorProps(object):

    def __init__(self, dataset, mtf_lsf):
        self.dataset = dataset
        self.mtf_lsf = mtf_lsf
        self.distorted_chip_data = self.dataset['blurred_chips']
        self.vector_data = self.vectorize_props()
        self.parse_keys = {'std', 'mtf', 'rer'}

    def vectorize_props(self):

        vector_data = {}
        for i, (chip_id, chip_data) in enumerate(self.distorted_chip_data.items()):

            measured_props = self.mtf_lsf[chip_id]

            if i == 0:
                for key, val in chip_data.items():
                    vector_data[key] = [val]
                for key, val in measured_props.items():
                    vector_data[key] = [val]
            else:
                for key, val in chip_data.items():
                    vector_data[key].append(val)
                for key, val in measured_props.items():
                    vector_data[key].append(val)

        return vector_data

    def parse_vector_data(self):

        parsed_vector_data = {}

        native_blur_vector = np.asarray(self.vector_data['native_blur'])
        native_blur_values = np.unique(native_blur_vector)
        kernel_sizes = np.unique(self.vector_data['kernel_size'])
        assert len(kernel_sizes) == 1

        for i, native_blur_value in enumerate(native_blur_values):

            for key, val in self.vector_data.items():

                if i == 0:
                    vector = np.asarray(val)
                    parsed_vector_data[key] = val
                else:
                    pass


def load_measured_mtf_lsf(directory):
    with open(Path(directory, 'mtf_lsf.json'), 'r') as file:
        data = json.load(file)
    return data


# def scale_freq_axis(f, resample_factor):
#     return resample_factor * f

#
# def get_nyquist(f_oversampled, oversample_factor):
#
#     num_freqs_oversampled = len(f_oversampled)
#     idx_nyquist = int(num_freqs_oversampled / oversample_factor)
#     f_nyquist = f_oversampled[idx_nyquist]
#
#     return idx_nyquist, f_nyquist

#
# def truncate(f, mtf, f_cut=0.5):
#
#     f_keep = f[np.where(f <= f_cut)]
#     mtf_keep = mtf[np.where(f <= f_cut)]
#
#     return f_keep, mtf_keep


def esf_extract(dataset, mtf_lsf_data):

    vector_props = VectorProps(dataset, mtf_lsf_data)

    return vector_props


if __name__ == '__main__':

    _directory_key = '0010'
    _kernel_size = 31
    _directory, _dataset = load_dataset(_directory_key, _kernel_size)
    _mtf_lsf_data = load_measured_mtf_lsf(_directory)

    _vector_props = esf_extract(_dataset, _mtf_lsf_data)

