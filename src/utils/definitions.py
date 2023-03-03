import os
import numpy as np
from yaml import safe_load
from pathlib import Path

"""
Contains global constants. 

Additional note: throughout project code, dataset generally refers to dataset metadata rather than the image files 
themselves. The dataset effectively records relevant metadata and relative paths to image files. This approach 
simplifies integration with W&B and allows quick artifact downloads, where the artifacts themselves are pointers
to local image files. 

"""

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))

STANDARD_DATASET_FILENAME = 'dataset.json'
STANDARD_CHECKPOINT_FILENAME = 'model_cp.pt'
STANDARD_BEST_LOSS_FILENAME = 'best_loss.pt'
STANDARD_TEST_RESULT_FILENAME = 'test_result.json'
STANDARD_CONFIG_USED_FILENAME = 'config_used.yml'
STANDARD_ENTROPY_PROPERTIES_FILENAME = 'entropy_properties.json'
STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME = 'effective_entropy_properties.json'
STANDARD_PROCESSED_DISTORTION_PERFORMANCE_PROPS_FILENAME = 'processed_distortion_performance_props.npz'
STANDARD_COMPOSITE_DISTORTION_PERFORMANCE_PROPS_FILENAME = 'composite_distortion_performance_props.npz'
STANDARD_PERFORMANCE_PREDICTION_FILENAME = 'performance_prediction_3d.npz'
STANDARD_UID_FILENAME = 'uid.json'
STANDARD_FIT_STATS_FILENAME = 'fit_stats.yml'
KEY_LENGTH = 4
STANDARD_TO_ORIGINAL_PAPER_LABEL_MAP_FILENAME = 'std_to_original_paper_coco_label_mappings.yml'
YOLO_TO_ORIGINAL_PAPER_LABEL_MAP_FILENAME = 'yolo_to_original_paper_coco_label_mappings.yml'

# defines standard paths in project structure for different artifact types
REL_PATHS = {
    'train_dataset': r'datasets/train',
    'test_dataset': r'datasets/test',
    'demo_dataset': r'datasets/demo',
    'images': r'images',
    'train_vectors': r'train_split',
    'val_vectors': r'val_split',
    'model': r'models',
    'test_result': r'test_results',
    'analysis': r'analysis',
    'distortion_scan': 'distortion_scans',
    'entropy': r'entropy',
    'rgb': r'rgb',
    'pan': r'0-pan',
    'res': r'1-res',
    'blur': r'2-blur',
    'noise': r'3-noise',
    'project_config': r'project_config',
    'perf_correlations': r'analysis/multi_result/perf_correlations',
    'pairwise': 'pairwise',
    '_extracted_artifact_props': r'analysis/_extracted_artifact_props',
    'composite_performance': r'analysis/composite_performance',
    'edges': 'edges',
    'rer_study': 'rer_study',
    'edge_chips': 'edge_chips',
    'demo_images': r'demo_images',
    'multi_result': r'analysis/multi_result',
    'bar_charts': r'analysis/bar_charts',
    'mosaics': r'mosaics',
    'composite_performance_configs': r'image-quality-for-deep-learning/src/analysis/composite_performance_configs',
    'noise_study': 'analysis/noise_study',
    'perf_prediction': 'perf_prediction',

    'label_definition_files':  r'image-quality-for-deep-learning/src/utils/label_definition_files',
    'temp_yaml':  r'image-quality-for-deep-learning/src/train/temp_yaml',
    'yolo_train_default_output_subdir': 'train',
    'yolo_val_default_output_subdir': 'val',
    'yolo_best_weights': 'weights/best.pt',
    'yolo_last_weights': 'weights/last.pt',

    'dataset_sub_struct': {  # keys correct to dataset_split_key variable
        'train': ('images/train', 'labels/train'),
        'val': ('images/val', 'labels/val'),
        'test': ('images/test', 'labels/test'),

    }
}

_project_config_filename = 'project_config.yml'
with open(Path(ROOT_DIR, REL_PATHS['project_config'], _project_config_filename), 'r') as file:
    _config = safe_load(file)  #
WANDB_PID = _config['PROJECT_ID']
NATIVE_RESOLUTION = _config['NATIVE_RESOLUTION']  # kept in config file so can be used in multi-project functions

DISTORTION_RANGE = {
    'sat6': {
        'res': (7, 28),  # where used, include check to ensure high end matches NATIVE_RESOLUTION
        'blur': (11, 0.1, 1.5),  # (kernel size, sigma_min, sigma_max)
        'noise': (0, 50)
    },
    'places365': {
        'res': (0.1, 1),  # not units specified differently btw sat6 and places
        'blur': (31, 0.1, 5),  # (kernel size, sigma_min, sigma_max)
        'noise': (0, 50)
    }
}

DISTORTION_RANGE_90 = {
    'sat6': {
        'res': (7, 28),  # where used, include check to ensure high end matches NATIVE_RESOLUTION
        'blur': (11, 0.5, 1.5),  # (kernel size, sigma_min, sigma_max)
        'noise': (0, 50)
    },
    'places365': {
        'res': (0.2, 1),  # not units specified differently btw sat6 and places
        'blur': (31, 0.5, 4.5),  # (kernel size, sigma_min, sigma_max)
        'noise': (0, 44)
    }
}
# the only change in the sat6 "90%" distortion range is in blur to mitigate the effects of changing blur std when
# the entire non-zero portion of the kernel falls within a single pixel

DISTORTION_TYPES = ['pan', 'res', 'blur', 'noise']

# ARTIFACT_TYPE_TAGS used to avoid name wandb name collisions
ARTIFACT_TYPE_TAGS = {
    'train_dataset': 'trn',
    'test_dataset': 'tst',
    'model': r'mdl',
    'test_result': r'rlt',
    'entropy_properties': r'ent',
    'demo_dataset': r'demo'
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

YOLO_TO_ORIGINAL_PAPER_KEY_MAPPING = None  # mappings loaded if WANDB_PID == 'coco'

if WANDB_PID[:4] == 'sat6':
    PROJECT_ID = 'sat6'
    # note: val slice in original datasets used for test dataset in this
    # project. Train datasets to have their own val slice carved out.
    ORIGINAL_DATASETS = {
        'sat6_full': {
            'rel_path': r'datasets/original',
            'metadata_filename': 'sat-6-full.mat',
            'artifact_type': 'full_dataset'
        },
    }

    # use 'model_file_config' as a standard for saving all models to enable easy loading
    ORIGINAL_PRETRAINED_MODELS = {
        'resnet18_sat6': {
            'model_file_config': {
                'model_rel_dir': r'none',  # in torchvsion.models library
                'model_filename': r'none', # stored as string to avoid error in load_pretrained_model()
            },
            'arch': 'resnet18_sat6',
            'artifact_type': 'model'
        },
        'resnet50_sat6': {
            'model_file_config': {
                'model_rel_dir': r'none',  # in torchvsion.models library
                'model_filename': r'none',  # stored as string to avoid error in load_pretrained_model()
            },
            'arch': 'resnet50_sat6',
            'artifact_type': 'model'
        },
        'densenet161_sat6': {
            'model_file_config': {
                'model_rel_dir': r'none',  # in torchvsion.models library
                'model_filename': r'none',  # stored as string to avoid error in load_pretrained_model()
            },
            'arch': 'densenet161_sat6',
            'artifact_type': 'model'
        }
    }

    NUM_CLASSES = 6

elif WANDB_PID[:6] == 'places':

    PROJECT_ID = 'places365'

    ORIGINAL_DATASETS = {
        'val_256': {
            'rel_path': r'datasets/test/val_256',
            'metadata_filename': 'places365_val.txt',
            'artifact_type': 'test_dataset'
        },
        'train_256_standard': {
            'rel_path': r'datasets/train/data_256',
            'metadata_filename': 'places365_train_standard.txt',
            'artifact_type': 'train_dataset'
        },
        'train_256_challenge': {
            'rel_path': r'datasets/train/challenge/data_256',
            'metadata_filename': 'places365_train_challenge.txt',
            'artifact_type': 'train_dataset'
        },
        'upsplash_demo': {
            'rel_path': r'datasets/demo/upsplash_256',
            'metadata_filename': 'upsplash_demo_256.txt',
            'artifact_type': 'demo_dataset'
        }
    }

    ORIGINAL_PRETRAINED_MODELS = {
        'resnet18_places365_as_downloaded': {
            'model_file_config': {
                'model_rel_dir': r'models/resnet18',
                'model_filename': 'resnet18_places365.pth.tar',
            },
            'arch': 'resnet18',
            'artifact_type': 'model'
        },
        'resnet50_places365_as_downloaded': {
            'model_file_config': {
                'model_rel_dir': r'models/resnet50',
                'model_filename': 'resnet50_places365.pth.tar',
            },
            'arch': 'resnet50',
            'artifact_type': 'model'
        },
        'densenet161_places365_as_downloaded': {
            'model_file_config': {
                'model_rel_dir': r'models/densenet161',
                'model_filename': 'densenet161_places365.pth.tar',
            },
            'arch': 'densenet161',
            'artifact_type': 'model'
        },
    }

    NUM_CLASSES = 365

elif WANDB_PID == 'coco':

    PROJECT_ID = 'coco'

    ORIGINAL_DATASETS = {
        'val2017': {
            'rel_path': r'datasets/test/val2017',
            'metadata_filename': 'instances_val2017.json',  # need to deal with conversion to "instances" nomenclature
            'artifact_type': 'test_dataset',
        },
        'train2017': {
            'rel_path': r'datasets/train/coco/images/train2017',
            'metadata_filename': 'instances_train2017.json',  # need to deal with conversion to "instances" nomenclature
            'artifact_type': 'train_dataset',

            'yolo_cfg': {
                'rel_path': r'datasets/train/coco/', # joined with ROOT_DIR for path in _data.yaml
                'train': 'train2017.txt',
                'val': 'val2017.txt',
            }
        },
        'coco128': {
            'rel_path': r'datasets/train/coco128/images/train2017',  # image directory, standard form across project
            'metadata_filename': 'dataset.json',
            'artifact_type': 'train_dataset',
            'yolo_cfg': {  # used to get data in the preferred yolo format
                'rel_path': r'datasets/train/coco128',  # joined with ROOT_DIR for path in _data.yaml
                'train': r'images/train2017',
                'val': r'images/train2017'
            }
        }
    }

    ORIGINAL_PRETRAINED_MODELS = {
        'fasterrcnn_resnet50_fpn': {
            'model_file_config': {
                'model_rel_dir': '',  # in torchvsion.models.detection library
                'model_filename': '',  # blank string to avoid throwing off path handling functions downstream
            },
            'arch': 'fasterrcnn',
            'artifact_type': 'model'
        },
        'yolov8n': {
            'model_file_config': {
                'model_rel_dir': '',  # in ultralytics library
                'model_filename': '',  # blank string to avoid throwing off path handling functions downstream
            },
            'arch': 'yolo',
            'artifact_type': 'model'
        },
        'yolov8m': {
            'model_file_config': {
                'model_rel_dir': '',  # in ultralytics library
                'model_filename': '',  # blank string to avoid throwing off path handling functions downstream
            },
            'arch': 'yolo',
            'artifact_type': 'model'
        },
        'yolov8l': {
            'model_file_config': {
                'model_rel_dir': '',  # in ultralytics library
                'model_filename': '',  # blank string to avoid throwing off path handling functions downstream
            },
            'arch': 'yolo',
            'artifact_type': 'model'
        },

    }

    NUM_CLASSES = None

    OBSOLETE_CLASS_IDS = {  # class IDs used in the original COCO paper, not included in 2014/2017 datasets
        12, # street sign
        26, # hat
        29, # shoe
        30, # eye_glasses
        45, # plate
        66, # mirror
        68, # window
        69, # desk
        71, # door
        83, # blender
        91, # hair brush
    }

    def get_yolo_to_original_key_mapping(call_count=0):

        if call_count > 2:
            raise Exception('Something has gone wrong with logging/accessing yolo_to_original_key_mapping')

        try:
            with open(Path(ROOT_DIR, REL_PATHS['project_config'], YOLO_TO_ORIGINAL_PAPER_LABEL_MAP_FILENAME), 'r') as f:
                data = safe_load(f)
            return data

        except FileNotFoundError:
            from src.utils.coco_label_functions import log_yolo_to_original_mapping
            log_yolo_to_original_mapping()

            call_count += 1

            return get_yolo_to_original_key_mapping(call_count=call_count)

    YOLO_TO_ORIGINAL_PAPER_KEY_MAPPING = get_yolo_to_original_key_mapping()


else:
    raise Exception('Invalid WANDB_PID')