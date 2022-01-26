import h5py
from pathlib import Path
import numpy as np
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS, ORIGINAL_DATASETS, STANDARD_DATASET_FILENAME
import json
from PIL import Image


def make_load_hdf5(image_dir, image_names, target_dir, num_load):

    for i in range(num_load):
        img_name = image_names[i][0]
        name_no_ext = img_name.split('.')[0]
        img = Image.open(Path(image_dir, img_name))
        img = np.asarray(img, dtype=np.uint8)
        file = h5py.File(target_dir / f'{name_no_ext}.h5', 'w')
        dataset_h5 = file.create_dataset(
            'image', np.shape(img), h5py.h5t.STD_U8BE, data=img
        )
        meta_set = file.create_dataset(
            'meta', np.shape(i), h5py.h5t.STD_U8BE, data=i
        )
        file.close()


if __name__ == '__main__':

    dataset_id = 'val_256'
    dataset_info = ORIGINAL_DATASETS[dataset_id]
    dataset_rel_path = dataset_info['rel_path']
    # TODO: can I use a dictionary to point to the relevant slices of the hdf5 file?

    dataset_abs_dir = Path(ROOT_DIR, dataset_rel_path)

    with open(Path(dataset_abs_dir, STANDARD_DATASET_FILENAME), 'r') as f:
        dataset = json.load(f)

    image_rel_dir = dataset['img_dir']
    image_abs_dir = Path(dataset_abs_dir, image_rel_dir)

    names_labels = dataset['names_labels']
    hdf5_dir = Path(ROOT_DIR, 'zz_sandbox', 'hdf5_test')
    if not hdf5_dir.exists():
        Path.mkdir(hdf5_dir)

    make_load_hdf5(image_abs_dir, names_labels, hdf5_dir, 5)
