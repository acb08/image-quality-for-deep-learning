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
PROJECT_ID = 'chip_distortions'
STANDARD_DATASET_FILENAME = 'dataset.json'
STANDARD_CHECKPOINT_FILENAME = 'model_cp.pt'
STANDARD_BEST_LOSS_FILENAME = 'best_loss.pt'
STANDARD_TEST_RESULT_FILENAME = 'test_result.json'
STANDARD_CONFIG_USED_FILENAME = 'config_used.yml'
STANDARD_ENTROPY_PROPERTIES_FILENAME = 'entropy_properties.json'
STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME = 'effective_entropy_properties.json'
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
    'analysis': r'analysis',
    'entropy': r'entropy',
    'pan': r'0-pan',
    'res': r'1-res',
    'blur': r'2-blur',
    'noise': r'3-noise',
    'mtf_study': r'mtf_study',
    'edge_chips': r'edge_chips',
    'edges': r'edges',
    'mtf': r'mtf',
    'distorted_edge_chips': r'distorted_edge_chips',
    'post_dist_props': r'post_dist_props'
}

DISTORTION_TYPES = ['pan', 'res', 'blur', 'noise']

# ARTIFACT_TYPE_TAGS used to avoid name wandb name collisions
ARTIFACT_TYPE_TAGS = {
    'train_dataset': 'trn',
    'test_dataset': 'tst',
    'model': r'mdl',
    'test_result': r'rlt',
    'entropy_properties': r'ent'
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

FIRST_DISC_ENCIRCLED_ENERGY = 0.838
AIRY_GAUSS_STD_EQUIV = 1.91
