import scipy.io
from pathlib import Path
from src.utils.definitions import ROOT_DIR, ORIGINAL_DATASETS

dataset_path = Path(ROOT_DIR, ORIGINAL_DATASETS['sat6_full']['rel_path'],
                    ORIGINAL_DATASETS['sat6_full']['metadata_filename'])
dataset = scipy.io.loadmat(dataset_path)

train_x, train_y = dataset['train_x'], dataset['train_y']
test_x, test_y = dataset['test_x'], dataset['test_y']
