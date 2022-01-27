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
PROJECT_ID = 'places_dry_run'
STANDARD_DATASET_FILENAME = 'dataset.json'
KEY_LENGTH = 4

# defines standard paths in project structure for different artifact types
REL_PATHS = {
    'train_dataset': r'datasets/train',
    'test_dataset': r'datasets/test',
    'images': r'images',
    'train_vectors': r'train_split',
    'val_vectors': r'val_split',
    'metadata': r'metadata',
    'model': r'models',
    'test_result': r'test_results',
    'analysis': r'analysis'
}

# note: val slice in original datasets used for test dataset in this
# project. Train datasets to have their own val slice carved out.
ORIGINAL_DATASETS = {
    'val_256': {
        'rel_path': r'datasets/test/val_256',
        'names_labels_filename': 'places365_val.txt',
        'artifact_type': 'test_dataset'
    },
    'train_256_standard': {
        'rel_path': r'datasets/train/data_256',
        'names_labels_filename': 'places365_train_standard.txt',
        'artifact_type': 'train_dataset'
    }
    # TODO: add train_256_challenge
}

# use 'model_file_config' as a standard for saving all models to enable easy loading
ORIGINAL_PRETRAINED_MODELS = {
    'resnet18_places365': {
        'model_file_config': {
            'model_rel_dir': r'models/resnet18',
            'filename': 'resnet18_places365.pth.tar',
        },
        'arch': 'resnet18',
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

# maps dataset type to its associated metadata filename located in the metadata dir
METADATA_FILENAMES = {
    'model': 'models.json',
    'train_dataset': 'train_datasets.json',
    'test_dataset': 'test_datasets.json',
    'test_result': 'test_results.json'
}
