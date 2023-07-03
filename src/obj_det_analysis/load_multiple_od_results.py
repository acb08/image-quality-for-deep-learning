"""
Placed here to avoid circular imports and changes in files that would result in version control driven
result recalculations
"""

import wandb
from src.obj_det_analysis.distortion_performance_od import get_obj_det_distortion_perf_result
from src.utils.definitions import WANDB_PID
from src.utils.functions import construct_artifact_id


def get_multiple_od_distortion_performance_results(result_id_pairs,
                                                   output_type='list',
                                                   distortions_ignore=(),
                                                   pre_processed_indices=None):
    if output_type == 'list':
        performance_results = []
    else:
        raise Exception('invalid output_type')

    if pre_processed_indices is None:
        pre_processed_indices = ()
    elif pre_processed_indices == 'all':
        pre_processed_indices = tuple(range(len(result_id_pairs)))

    with wandb.init(project=WANDB_PID, job_type='analyze_test_result') as run:

        for i, (artifact_id, identifier) in enumerate(result_id_pairs):

            if i in pre_processed_indices:
                pre_processed_artifact = True
            else:
                pre_processed_artifact = False

            artifact_id, __ = construct_artifact_id(artifact_id)
            distortion_performance_result, __ = get_obj_det_distortion_perf_result(
                result_id=artifact_id, identifier=identifier,
                make_dir=False,
                run=run,
                distortions_ignore=distortions_ignore,
                pre_processed_artifact=pre_processed_artifact
            )

            if output_type == 'list':
                performance_results.append(distortion_performance_result)

    return performance_results
