"""
Checks for changes to key portions of codebase used to calculate mAP and fit performance predictions so that I can
verify that cached files of expensive computational results are still valid (i.e. if I store mAP as a function of
resolution, blur, and noise for a particular model/dataset, I need to make sure that any changes to my mAP code trigger
a recalculation of the mAP result next time I try to use it.)
"""

from hashlib import blake2b
from pathlib import Path
from src.utils.definitions import ROOT_DIR

_CODE_ROOT = Path(ROOT_DIR, 'image-quality-for-deep-learning', 'src')

_mAP_CODE_PATHS = (
    'utils/shared_methods.py',
    'utils/definitions.py',
    'utils/detection_functions.py',
    'utils/classes.py',
    'obj_det_analysis/classes.py',
    'analysis/analysis_functions.py',
    'obj_det_analysis/analysis_tools.py'
)


def get_map_hash_mash():

    hash_mash = str()

    for code_path in _mAP_CODE_PATHS:

        with open(Path(_CODE_ROOT, code_path), 'r') as f:
            data = f.read()
            hash_val = blake2b(data.encode('utf-8')).hexdigest()
            hash_mash += hash_val

    return hash_mash


if __name__ == '__main__':

    _hash_mash = get_map_hash_mash()
    print(_hash_mash)
