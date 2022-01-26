import json
from pathlib import Path
from definitions import ROOT_DIR, REL_PATHS, METADATA_FILENAMES
from functions import potential_overwrite

if __name__ == '__main__':

    header = 'mini database to log artifacts in parallel to W&B logging'
    data = {'header': header}
    target_dir_rel_path = REL_PATHS['metadata']
    target_dir = Path(ROOT_DIR, target_dir_rel_path)

    for file_key in METADATA_FILENAMES:
        filename = METADATA_FILENAMES[file_key]
        file_path = Path(target_dir, filename)
        if not potential_overwrite(file_path):
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:
            print(file_path, 'already exists!')
