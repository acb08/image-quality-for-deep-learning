"""
Placed here to avoid circular imports and changes in files that would result in version control driven
result recalculations
"""

import wandb
from src.obj_det_analysis.distortion_performance_od import get_obj_det_distortion_perf_result
from src.utils.definitions import WANDB_PID
from src.utils.functions import construct_artifact_id


def get_multiple_od_distortion_performance_results(result_id_pairs,
                                                   output_type='list'):

    if output_type == 'list':
        performance_results = []
    else:
        raise Exception('invalid output_type')

    with wandb.init(project=WANDB_PID, job_type='analyze_test_result') as run:

        for artifact_id, identifier in result_id_pairs:

            artifact_id, __ = construct_artifact_id(artifact_id)
            distortion_performance_result, __ = get_obj_det_distortion_perf_result(result_id=artifact_id,
                                                                                   identifier=identifier,
                                                                                   make_dir=False,
                                                                                   run=run)

            if output_type == 'list':
                performance_results.append(distortion_performance_result)

    return performance_results
