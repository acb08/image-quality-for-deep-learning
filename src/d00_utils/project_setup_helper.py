from pathlib import Path
from src.d00_utils.definitions import REL_PATHS, ROOT_DIR, ORIGINAL_DATASETS


directory_keys = [
    'train_dataset',
    'test_dataset',
    'model',
    'test_result',
    'analysis',
]


def make_directories():

    for dir_key in directory_keys:
        target_directory = Path(ROOT_DIR, REL_PATHS[dir_key])
        if not target_directory.is_dir():
            Path.mkdir(target_directory, parents=True, exist_ok=True)


def check_for_original_dataset():

    all_datasets_found = True
    for key in ORIGINAL_DATASETS:

        dataset_info = ORIGINAL_DATASETS[key]
        full_path = Path(ROOT_DIR, dataset_info['rel_path'], dataset_info['names_labels_filename'])

        if not full_path.is_file():
            all_datasets_found = False
            print(f'{str(full_path)} not found')

    if not all_datasets_found:
        print('Original dataset not found in project structure')


if __name__ == '__main__':

    make_directories()
    check_for_original_dataset()

