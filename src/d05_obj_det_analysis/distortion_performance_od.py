from src.d04_analysis.distortion_performance import load_dataset_and_result
from src.d00_utils.classes import COCO
import numpy as np
import argparse
from pathlib import Path
from src.d00_utils.functions import get_config
from src.d00_utils import definitions
import wandb
from src.d05_obj_det_analysis.analyze import calculate_aggregate_results


class DistortedCOCODataset(COCO):

    def __init__(self, dataset, distortion_ids, convert_to_std, load_local=False):

        self._dataset = dataset
        self.distortion_ids = distortion_ids
        self.convert_to_std = convert_to_std
        self.load_local = load_local
        if self.load_local:
            raise NotImplementedError

        COCO.__init__(self,
                      image_directory=None,
                      instances=self._dataset['instances'],
                      transform=None)

    def build_distortion_vectors(self):
        """
        Pull out distortion info from self._dataset['images] and place in numpy vectors
        """
        pass

    def log_semi_processed_data(self):
        """
        Store distortion vectors in .json file to speed up working with
        """
        pass


def _fetch_obj_det_distortion_performance_result(run, result_id, identifier, distortion_ids, make_dir=True):

    output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['analysis'], result_id)
    if make_dir and not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    dataset, result, dataset_id = load_dataset_and_result(run=run, result_id=result_id)

    return dataset, result, dataset_id, output_dir


def get_obj_det_distortion_performance_result(result_id=None, identifier=None, config=None,
                                              distortion_ids=('res', 'blur', 'noise'), make_dir=True):

    if not result_id and not identifier:
        result_id = config['result_id']
        identifier = config['identifier']

    with wandb.init(project=definitions.WANDB_PID, job_type='analyze_test_result') as run:
        dataset, result, dataset_id, output_dir = _fetch_obj_det_distortion_performance_result(run, result_id,
                                                                                               identifier,
                                                                                               distortion_ids,
                                                                                               make_dir=make_dir)
    return dataset, result, dataset_id, output_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_analysis_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_analysis_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    _dataset, _result, _dataset_id, _output_dir = get_obj_det_distortion_performance_result(config=run_config)

    _outputs, _targets = _result['outputs'], _result['targets']
    calculate_aggregate_results(outputs=_outputs,
                                targets=_targets,
                                output_dir_abs=_output_dir)
