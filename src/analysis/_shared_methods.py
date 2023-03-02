"""
Functions intended to be called by methods of multiple classes that do not have inheritance or composition
relationships. I suspect a SW developer would disapprove, but it seemed to be the best way to manage commonality
between very similar bu crucially different classes.
"""
from pathlib import Path
# from src.analysis.distortion_performance import ModelDistortionPerformanceResult
import numpy as np
from hashlib import blake2b
from src.utils.definitions import ROOT_DIR, REL_PATHS, STANDARD_PROCESSED_DISTORTION_PERFORMANCE_PROPS_FILENAME
from src.analysis.analysis_functions import build_3d_field


def _get_processed_instance_props_path(_self, predict_eval_flag=None):
    props_dir = Path(ROOT_DIR, REL_PATHS['_extracted_artifact_props'], _self.result_id)
    if predict_eval_flag:
        props_dir = Path(props_dir, predict_eval_flag)
    if not props_dir.is_dir():
        Path.mkdir(props_dir, parents=True)
    props_path = Path(props_dir, STANDARD_PROCESSED_DISTORTION_PERFORMANCE_PROPS_FILENAME)
    return props_path


# def _check_limits_(values, limits):
#
#     if np.max(values) > np.max(limits):
#         return False
#     elif np.min(values) < np.min(limits):
#         return False
#     else:
#         return True


def _check_extract_processed_props(_self, predict_eval_flag=None,
                                   # res_limits=None,
                                   # blur_limits=None,
                                   # noise_limits=None
                                   ):

    processed_props_path = _self.get_processed_instance_props_path(predict_eval_flag=predict_eval_flag)
    if not Path.is_file(processed_props_path):
        return False

    processed_props = np.load(processed_props_path)
    processed_props_hash = processed_props['_instance_hash']
    if _self.instance_hashes[predict_eval_flag] != processed_props_hash:
        return False

    res_values = processed_props['res_values']
    # if res_limits:
    #     if not _check_limits_(res_values, res_limits):
    #         return False

    blur_values = processed_props['blur_values']
    # if blur_limits:
    #     if not _check_limits_(blur_values, blur_limits):
    #         return False

    noise_values = processed_props['noise_values']
    # if noise_limits:
    #     if not _check_limits_(noise_values, noise_limits):
    #         return False

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


def _get_3d_distortion_perf_props(_self, distortion_ids, predict_eval_flag='predict'):

    if distortion_ids != ('res', 'blur', 'noise'):
        raise ValueError('method requires distortion_ids (res, blur, noise)')

    existing_processed_props = _self.check_extract_processed_props(predict_eval_flag=predict_eval_flag)

    if existing_processed_props:
        print('loading existing processed properties')
        res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, __ = existing_processed_props
    else:
        if predict_eval_flag == 'predict':
            print('processing 3d props')
            res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(
                _self.res_predict, _self.blur_predict, _self.noise_predict, _self.top_1_vec_predict, data_dump=True)
            _self.archive_processed_props(res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array,
                                          predict_eval_flag)
        elif predict_eval_flag == 'eval':
            print('processing 3d props')
            res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, __ = build_3d_field(
                _self.res, _self.blur, _self.noise, _self.top_1_vec, data_dump=True)
            _self.archive_processed_props(res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array,
                                          predict_eval_flag)
        else:
            raise Exception('invalid predict_eval_flag')

    return res_values, blur_values, noise_values, perf_3d, distortion_array, perf_array, None


def get_instance_hash(performance_array, distortion_array):
    mega_string = str(performance_array) + str(distortion_array)
    return blake2b(mega_string.encode('utf-8')).hexdigest()
