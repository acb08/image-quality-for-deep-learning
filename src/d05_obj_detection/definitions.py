from pathlib import Path

ROOT_DIR = Path.cwd().parent
print(str(ROOT_DIR))
ANNOTATION_DIR = Path(ROOT_DIR, r'annotations')

DATASET_PATHS = {
    'val2017': {
        'instances': 'instances_val2017.json',
        'image_dir': 'val2017'
    }
}
