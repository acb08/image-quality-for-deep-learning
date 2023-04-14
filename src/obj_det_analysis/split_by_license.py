from pathlib import Path
import yaml
from src.utils.definitions import REL_PATHS, ROOT_DIR
import shutil


def split(directory, move_copy_flag='copy'):

    directory = Path(directory).expanduser()
    license_map = load_license_map()

    for filename, (license_num, url) in license_map.items():
        license_dir = make_check_license_dir(directory, license_num)
        if move_copy_flag == 'copy':
            shutil.copy(Path(directory, filename), Path(license_dir, filename))
        elif move_copy_flag == 'move':
            shutil.move(Path(directory, filename), Path(license_dir, filename))
        else:
            raise ValueError("move_copy_flag must be either 'move' or 'copy'")


def load_license_map():

    with open(Path(ROOT_DIR, REL_PATHS['project_config'], 'license_map_128.yml'), 'r') as f:
        license_map = yaml.safe_load(f)

    return license_map


def make_check_license_dir(directory, license_num):

    license_dir = Path(directory, f'cc-{license_num}')
    if not license_dir.is_dir():
        Path.mkdir(license_dir)

    return license_dir


if __name__ == '__main__':

    _directory = r'~/coco/datasets/demo/compare/mp90-image-chain'
    split(_directory)
    # print(load_license_map())
    # _license_map = load_license_map()
    # for key, value in _license_map.items():
    #     print(key, value)
