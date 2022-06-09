"""
Functions intended to be called by methods of multiple classes that do not have inheritance or composition
relationships.
"""
from pathlib import Path
# from src.d04_analysis.distortion_performance import ModelDistortionPerformanceResult
import numpy as np

from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_PROCESSED_DISTORTION_PERFORMANCE_PROPS_FILENAME
from src.d04_analysis.analysis_functions import build_3d_field


def _get_processed_instance_props_path(_self, predict_eval_flag=None):
    props_dir = Path(ROOT_DIR, REL_PATHS['_extracted_artifact_props'], _self.result_id)
    if predict_eval_flag:
        props_dir = Path(props_dir, predict_eval_flag)
    if not props_dir.is_dir():
        Path.mkdir(props_dir, parents=True)
    props_path = Path(props_dir, STANDARD_PROCESSED_DISTORTION_PERFORMANCE_PROPS_FILENAME)
    return props_path


def _check_extract_processed_props(_self, predict_eval_flag=None):

    processed_props_path = _self.get_processed_instance_props_path(predict_eval_flag=predict_eval_flag)
    if not Path.is_file(processed_props_path):
        return False

    processed_props = np.load(processed_props_path)
    processed_props_hash = processed_props['_instance_hash']
    if _self.instance_hashes[predict_eval_flag] != processed_props_hash:
        return False

    res_values = processed_props['res_values']
    blur_values = processed_props['blur_values']
    noise_values = processed_props['noise_values']
    perf_3d = processed_props['perf_3d']
    distortion_array = processed_props['distortion_array']
    perf_array = processed_props['perf_array']

    return res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, None


def _archive_processed_props(_self, res_values, blur_values, noise_values, perf_3d, distortion_array,
                             perf_array, predict_eval_flag=None,
                             distortion_array_perf_predict=None, top_1_array_perf_predict=None,
                             top_1_array_eval=None):

    processed_props_path = _self.get_processed_instance_props_path(predict_eval_flag=predict_eval_flag)

    np.savez_compressed(processed_props_path,
                        _instance_hash=_self.instance_hashes[predict_eval_flag],
                        res_values=res_values,
                        blur_values=blur_values,
                        noise_values=noise_values,
                        perf_3d=perf_3d,
                        distortion_array=distortion_array,
                        perf_array=perf_array)


def _get_3d_distortion_perf_props(_self, distortion_ids, predict_eval_flag=None):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise ValueError('method requires distortion_ids (res, blur, noise)')

    if predict_eval_flag is None:
        predict_eval_flag = 'predict'

    if predict_eval_flag == 'predict':
        existing_processed_props = _self.check_extract_processed_props(predict_eval_flag=predict_eval_flag)
        if existing_processed_props:
            print('loading existing processed properties')
            res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, __ = existing_processed_props
        else:
            print('processing 3d props')
            res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(
                _self.res, _self.blur, _self.noise, _self.top_1_vec_predict, data_dump=True)
            _self.archive_processed_props(res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array,
                                          predict_eval_flag)

        return res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, None

    elif predict_eval_flag == 'eval':
        pass

    else:
        raise Exception("predict_eval_flag must be either 'predict' or 'eval'")
