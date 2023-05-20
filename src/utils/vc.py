"""
Checks for changes to key portions of codebase used to calculate mAP and fit performance predictions so that I can
verify that cached files of expensive computational results are still valid (i.e. if I store mAP as a function of
resolution, blur, and noise for a particular model/dataset, I need to make sure that any changes to my mAP code trigger
a recalculation of the mAP result next time I try to use it.)
"""

from hashlib import blake2b
from pathlib import Path
from src.utils.definitions import CODE_ROOT


_mAP_CODE_PATHS = (
    'src/utils/shared_methods.py',
    'src/utils/definitions.py',
    'src/utils/detection_functions.py',
    'src/utils/classes.py',
    'src/obj_det_analysis/classes.py',
    'src/analysis/analysis_functions.py',
    'src/obj_det_analysis/analysis_tools.py'
)

_DISTORTION_PERFORMANCE_COMPOSITE_OD_PATHS = (
    'distortion_performance_composite_od.py',
)


def get_map_hash_mash(_code_paths=_mAP_CODE_PATHS):

    hash_mash = str()

    for code_path in _mAP_CODE_PATHS:

        with open(Path(CODE_ROOT, code_path), 'r') as f:
            data = f.read()
            hash_val = blake2b(data.encode('utf-8')).hexdigest()
            hash_mash += hash_val

    return hash_mash


def get_od_composite_hash_mash():
    paths = _mAP_CODE_PATHS + _DISTORTION_PERFORMANCE_COMPOSITE_OD_PATHS
    return get_map_hash_mash(_code_paths=paths)


if __name__ == '__main__':

    _hash_mash = get_map_hash_mash()
    print(_hash_mash)
