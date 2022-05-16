import wandb
from src.d00_utils.definitions import STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME, \
    STANDARD_ENTROPY_PROPERTIES_FILENAME, STANDARD_DATASET_FILENAME, PROJECT_ID, ROOT_DIR, REL_PATHS
from src.d00_utils.functions import get_config
from src.d04_analysis.distortion_performance import ModelDistortionPerformanceResult
from src.d04_analysis.distortion_entropy import DistortionEntropyProperties
from src.d04_analysis.distortion_entropy import analyze_entropy_3d
from src.d04_analysis.analysis_functions import build_3d_field
import argparse
from pathlib import Path
import numpy as np


class ModelDistortionEntropyPerformance(DistortionEntropyProperties, ModelDistortionPerformanceResult):

    def __init__(self, run, result_id, convert_to_std=True, identifier=None):

        self.convert_to_std = convert_to_std
        ModelDistortionPerformanceResult.__init__(self, run, result_id, convert_to_std=self.convert_to_std)
        # self.dataset_id inherited from ModelDistortionPerformance
        DistortionEntropyProperties.__init__(self, run, self.dataset_id, convert_to_std=self.convert_to_std)
        self.identifier = identifier
        self.properties_3d = self.get_properties_3d()
        self._entropy_performance_correlations = None

    def get_properties_3d(self):

        x_id, y_id, z_id = self.distortion_ids
        x = self.distortions[x_id]
        y = self.distortions[y_id]
        z = self.distortions[z_id]
        properties_3d = {}

        for effective_entropy_id, effective_entropy in self.effective_entropy_props.items():
            effective_entropy_3d = build_3d_field(x, y, z, effective_entropy, data_dump=False)
            properties_3d[effective_entropy_id] = effective_entropy_3d

        accuracy_3d = build_3d_field(x, y, z, self.top_1_vec, data_dump=False)
        properties_3d['mean_accuracy_3d'] = accuracy_3d

        return properties_3d

    def get_entropy_performance_correlations(self, re_calculate=False):

        if self._entropy_performance_correlations and not re_calculate:
            return self._entropy_performance_correlations

        correlations = {}
        mean_performance_vector = np.ravel(self.properties_3d['mean_accuracy_3d'])
        for effective_entropy_id in self.effective_entropy_ids:
            effective_entropy_3d = self.properties_3d[effective_entropy_id]
            correlation = np.corrcoef(np.ravel(effective_entropy_3d), mean_performance_vector)
            correlations[effective_entropy_id] = correlation

        self._entropy_performance_correlations = correlations

        return self._entropy_performance_correlations


def get_distortion_entropy_performance_result(result_id=None, identifier=None, convert_to_std=True, config=None):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']
        convert_to_std = config['convert_to_std']

    with wandb.init(project=PROJECT_ID, job_type='analyze_test_result') as run:
        output_dir = Path(ROOT_DIR, REL_PATHS['analysis'], result_id, REL_PATHS['entropy'])
        if not output_dir.is_dir():
            Path.mkdir(output_dir)

        model_distortion_entropy_perf = ModelDistortionEntropyPerformance(run, result_id, convert_to_std=convert_to_std,
                                                                          identifier=identifier)

    return model_distortion_entropy_perf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_entropy_performance_config.yml',
                        help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_entropy_performance_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    mdl_dist_entropy_perf = get_distortion_entropy_performance_result(config=run_config)

