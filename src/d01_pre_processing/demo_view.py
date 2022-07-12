from src.d00_utils.functions import load_wandb_data_artifact, load_npz_data
import src.d00_utils.definitions as definitions
import numpy as np
from PIL import Image
from pathlib import Path


def extract_to_png(image_array, label_array, output_dir, idx_offset=0):

    num, h, w, c = np.shape(image_array)

    for idx in range(num):
        image = image_array[idx]
        image = Image.fromarray(image)
        label = label_array[idx]
        filename = f'img_{idx + idx_offset}_{label}.png'
        image_path = Path(output_dir, filename)
        image.save(image_path)

    return idx


def np_dataset_to_png(input_directory, output_directory=None, filenames=None):

    if not filenames:
        filenames = get_npz_files(input_directory)

    if not output_directory:
        output_directory = get_corresponding_output_dir(input_directory)

    idx_offset = 0

    for filename in filenames:

        data = load_npz_data(input_directory, filename)
        image_array = data['images']
        label_array = data['labels']

        idx_offset += extract_to_png(image_array, label_array, output_directory, idx_offset=idx_offset)

    return output_directory


def get_corresponding_output_dir(input_directory, mkdir=True):
    input_directory = Path(input_directory)
    sub_dir = Path(input_directory.parts[-2], input_directory.parts[-1])
    output_directory = Path(definitions.ROOT_DIR, definitions.REL_PATHS['demo_images'], sub_dir)
    if mkdir:
        output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def get_npz_files(directory):
    directory = Path(directory)
    filenames = list(directory.iterdir())
    filenames = [str(filename.parts[-1]) for filename in filenames]
    npz_filenames = [filename for filename in filenames if filename[-3:] == 'npz']
    return npz_filenames


def combine(images, strip_shape, background):

    image_strip = background * np.ones(strip_shape, dtype=np.uint8)
    horizontal_offset = 0
    for image in images:
        h, w, __ = np.shape(image)
        image_strip[:h, horizontal_offset:horizontal_offset + w, :] = image
        horizontal_offset += w

    return image_strip


def make_image_strips(source_directories, output_directory, extension='png', background=255):

    base_directory = Path(source_directories[0])

    image_filenames = list(base_directory.iterdir())
    image_filenames = [str(filename.parts[-1]) for filename in image_filenames]
    image_filenames = [filename for filename in image_filenames if filename[-len(extension):] == extension]

    strip_shape = None
    height, width, channels = 0, 0, 3

    for i, image_filename in enumerate(image_filenames):
        images = []
        for j, directory in enumerate(source_directories):
            image = Image.open(Path(directory, image_filename))
            image = np.asarray(image)
            images.append(image)
            if i == 0:
                h, w, __ = np.shape(image)
                height = max(height, h)
                width += w

        if i == 0:
            strip_shape = (height, width, channels)

        image_strip = combine(images, strip_shape, background)
        output_path = Path(output_directory, image_filename)
        image_strip = Image.fromarray(image_strip)
        image_strip.save(output_path)


def main(input_directories, image_strip_directory=None):

    output_directories = []

    for input_directory in input_directories:
        output_directory = np_dataset_to_png(input_directory)
        output_directories.append(output_directory)

    if image_strip_directory:
        if not image_strip_directory.is_dir():
            image_strip_directory.mkdir(parents=True, exist_ok=True)
        make_image_strips(output_directories, image_strip_directory)


if __name__ == '__main__':

    # _input_directories = [
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/rgb',
    #     r'/home/acb6595/places/datasets/test/0001-tst-mp90_demo/0-pan',
    #     r'/home/acb6595/places/datasets/test/0001-tst-mp90_demo/3-noise',
    #     r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/3-noise'
    #     ]

    # _input_directories = [
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/rgb',
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/0-pan',
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/1-res',
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/2-blur',
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/3-noise',
    #     ]

    _input_directories = [
        r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/rgb',
        r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/0-pan',
        r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/1-res',
        r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/2-blur',
        r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/3-noise',
    ]

    # _output_directory_name = 'mp90_image_chain_rgb'
    # _output_directory_name = 'rgb_origin_mp90_ep90'
    _output_directory_name = 'ep90_image_chain_rgb'

    _image_strip_output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['demo_images'], _output_directory_name)

    main(_input_directories, image_strip_directory=_image_strip_output_dir)

