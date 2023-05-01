"""
Used to log distortion performance properties as a wandb artifact when test result is too large to process locally
"""

from src.obj_det_analysis.distortion_performance_od import get_obj_det_distortion_perf_result
from src.obj_det_analysis.classes import ModelDistortionPerformanceResultOD
from src.utils.functions import get_config
from src.utils.definitions import WANDB_PID
import argparse
from pathlib import Path
import wandb


def _get_processed_props_artifact_id(result_id):
    result_id = str(result_id)
    return result_id.replace(':', '-')


def log_props_artifact(config):

    wandb.login()

    with wandb.init(project=WANDB_PID, job_type='archive_performance_properties') as run:

        distortion_performance_result, output_dir = get_obj_det_distortion_perf_result(config=run_config,
                                                                                       run=run)

        distortion_performance_result.get_3d_distortion_perf_props()
        props_path = distortion_performance_result.get_processed_instance_props_path()
        result_id = distortion_performance_result.result_id
        props_artifact_id = _get_processed_props_artifact_id(result_id)

        new_artifact = wandb.Artifact(props_artifact_id,
                                      type='processed_performance_properties',
                                      metadata=config)
        new_artifact.add_file(props_path)
        run.log_artifact(new_artifact)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_analysis_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_analysis_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    log_props_artifact(run_config)
