import os
import numpy as np

"""
Contains global constants. 

Additional note: throughout project code, dataset generally refers to dataset metadata rather than the image files 
themselves. The dataset effectively records relevant metadata and relative paths to image files. This approach 
simplifies integration with W&B and allows quick artifact downloads, where the artifacts themselves are pointers
to local image files. 

"""

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))
PROJECT_ID = 'sat6'
STANDARD_DATASET_FILENAME = 'dataset.json'
STANDARD_CHECKPOINT_FILENAME = 'model_cp.pt'
STANDARD_BEST_LOSS_FILENAME = 'best_loss.pt'
STANDARD_TEST_RESULT_FILENAME = 'best_result.json'
STANDARD_CONFIG_USED_FILENAME = 'config_used.yml'
KEY_LENGTH = 4

# defines standard paths in project structure for different artifact types
REL_PATHS = {
    'train_dataset': r'datasets/train',
    'test_dataset': r'datasets/test',
    'images': r'images',
    'train_vectors': r'train_split',
    'val_vectors': r'val_split',
    'model': r'models',
    'test_result': r'test_results',
    'analysis': r'analysis'
}

# ARTIFACT_TYPE_TAGS used to avoid name wandb name collisions
ARTIFACT_TYPE_TAGS = {
    'train_dataset': 'trn',
    'test_dataset': 'tst',
    'model': r'mdl',
    'test_result': r'rlt'
}

# note: val slice in original datasets used for test dataset in this
# project. Train datasets to have their own val slice carved out.
ORIGINAL_DATASETS = {
    'sat6_full': {
        'rel_path': r'datasets/original',
        'names_labels_filename': 'sat-6-full.mat',
        'artifact_type': 'full_dataset'
    },
}

# use 'model_file_config' as a standard for saving all models to enable easy loading
ORIGINAL_PRETRAINED_MODELS = {
    'resnet18_places365_as_downloaded': {
        'model_file_config': {
            'model_rel_dir': r'models/resnet18',
            'model_filename': 'resnet18_places365.pth.tar',
        },
        'arch': 'resnet18',
        'artifact_type': 'model'
    },
    'resnet18_sat6': {
        'model_file_config': {
            'model_rel_dir': r'none',  # in torchvsion.models library
            'model_filename': r'none', # stored as string to avoid error in load_pretrained_model()
        },
        'arch': 'resnet18_sat6',
        'artifact_type': 'model'
    }
}


# enables storing data types as strings that are json serializable
DATATYPE_MAP = {
    'np.uint8': np.uint8,
    'np.float32': np.float32,
    'np.float64': np.float64,
    'np.int16': np.int16,
    'np.int32': np.int32,
    'np.int64': np.int64,
}

# removing redundant project metadata
# maps dataset type to its associated metadata filename located in the metadata dir
# METADATA_FILENAMES = {
#     'model': 'models.json',
#     'train_dataset': 'train_datasets.json',
#     'test_dataset': 'test_datasets.json',
#     'test_result': 'test_results.json'
# }
