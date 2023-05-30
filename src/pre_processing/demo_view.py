from src.utils.functions import load_wandb_data_artifact, load_npz_data
import src.utils.definitions as definitions
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


def combine_horizontal(images, strip_shape, background):

    image_strip = background * np.ones(strip_shape, dtype=np.uint8)
    horizontal_offset = 0
    for image in images:
        h, w, __ = np.shape(image)
        image_strip[:h, horizontal_offset:horizontal_offset + w, :] = image
        horizontal_offset += w

    return image_strip


def _get_filenames(directory, extension, exclude=None):

    directory = Path(directory)

    image_filenames = list(directory.iterdir())
    image_filenames = [str(filename.parts[-1]) for filename in image_filenames]
    image_filenames = [filename for filename in image_filenames if filename[-len(extension):] == extension]

    if exclude:
        if type(exclude) not in (tuple, list):
            exclude = [exclude]
        for item in exclude:
            if item in image_filenames:
                image_filenames.remove(item)

    return image_filenames


def make_image_strips_multi_dir(source_directories, output_directory, extension='png', background=255,
                                base_directory_idx=0):

    """
    Concatenates different versions of an image into a horizontal strip, where corresponding image versions are
    segregated by directory
    """

    base_directory = Path(source_directories[base_directory_idx])

    image_filenames = _get_filenames(base_directory, extension)

    strip_shape = None

    for i, image_filename in enumerate(image_filenames):
        images = []
        height, width, channels = 0, 0, 0
        for j, directory in enumerate(source_directories):
            image = Image.open(Path(directory, image_filename))
            image = image.convert('RGB')
            image = np.asarray(image)
            images.append(image)
            # if j == 0:
            h, w, channels = np.shape(image)
            height = max(height, h)
            width += w

        # if i == 0:
        strip_shape = (height, width, channels)

        image_strip = combine_horizontal(images, strip_shape, background)
        output_path = Path(output_directory, image_filename)
        image_strip = Image.fromarray(image_strip)
        image_strip.save(output_path)


def make_mosaics_multi_dir(source_directories, output_directory=None, extension='png', side_len=8, num_mosaics=1):

    """
    Makes mosaics using different versions of a set of images, where corresponding image versions are segregated by
    directory
    """

    base_directory = Path(source_directories[0])
    if not output_directory:
        output_directory = Path(base_directory.parents[0], 'mosaics')
    image_filenames = _get_filenames(base_directory, extension)

    images_per_mosaic = side_len ** 2
    total_images = num_mosaics * images_per_mosaic
    if total_images > len(image_filenames):
        raise ValueError('not enough images for the number size and number of mosaics specified')

    if num_mosaics != 1:
        raise NotImplementedError('only single mosaics implemented')

    for i in range(num_mosaics):
        mosaic_filenames = image_filenames[i * images_per_mosaic: (i + 1) * images_per_mosaic]
        for j, directory in enumerate(source_directories):
            source_id = str(Path(directory).parts[-1])
            output_mosaic_filename = f'{j}_mosaic_{source_id}.png'
            make_mosaic(directory, image_filenames=mosaic_filenames, output_filename=output_mosaic_filename,
                        output_dir=output_directory)

    return output_directory


def make_mosaic(source_directory, image_filenames=None, output_dir=None, extension='png', side_len=8,
                output_filename='mosaic.png'):

    if not image_filenames:
        image_filenames = _get_filenames(source_directory, exclude=output_filename, extension=extension)

    if not output_dir:
        output_dir = source_directory

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    if len(image_filenames) != side_len ** 2:
        print('Warning: number of image filenames should be equal to side_len ** 2')

    strips = []

    for i in range(side_len):
        row_filenames = image_filenames[i * side_len: (i + 1) * side_len]
        strip = make_horizontal_strip(source_directory, image_filenames=row_filenames, return_strip=True)
        strips.append(strip)

    make_vertical_stack(None, images=strips, output_dir=output_dir, output_filename=output_filename)


def make_vertical_stack(directory, image_filenames=None, extension='png', output_dir=None, output_filename='stack.png',
                        images=None):

    if images is not None and image_filenames is not None:
        raise Exception('Function received both images and image filenames. Please make up your mind :)')

    if not image_filenames and not images:
        image_filenames = _get_filenames(directory, extension, exclude=output_filename)

    if not images:
        images = []
        open_from_dir = True
    else:  # hideous hack to let this function work with either images or a list of filenames
        image_filenames = [f'dummy_filename{i}' for i in range(len(images))]
        open_from_dir = False

    for i, filename in enumerate(image_filenames):
        if open_from_dir:
            image = Image.open(Path(directory, filename))
            image = np.asarray(image, dtype=np.uint8)
            images.append(image)
        else:
            image = images[i]
        if i == 0:
            strip_shape = np.shape(image)
        if i != 0:
            if strip_shape != np.shape(image):
                raise Exception('All image strips must be of identical shape to stack')

    h, w, c = strip_shape
    h_stack = h * len(images)
    stack = np.zeros((h_stack, w, c), dtype=np.uint8)

    for i, image in enumerate(images):
        vertical_offset = i * h
        stack[vertical_offset: vertical_offset + h, :, :] = image

    if not output_dir:
        output_dir = directory

    stack = Image.fromarray(stack)
    stack.save(Path(output_dir, output_filename))


def make_horizontal_strip(directory, image_filenames=None, extension='png', output_dir=None,
                          output_filename='strip.png', background=255, return_strip=False):

    if not image_filenames:
        image_filenames = _get_filenames(directory, extension, exclude=output_filename)
        image_filenames = sorted(image_filenames)

    images = []

    height = None
    width = None
    c = None

    for i, filename in enumerate(image_filenames):
        image = Image.open(Path(directory, filename))
        image = np.asarray(image, dtype=np.uint8)
        images.append(image)
        h, w, c = np.shape(image)
        if i == 0:
            height = h
            width = w
        else:
            width += w
            height = max(height, h)

    strip_shape = (height, width, c)
    strip = combine_horizontal(images, strip_shape, background)

    if return_strip:
        return strip

    else:
        if not output_dir:
            output_dir = directory
        strip = Image.fromarray(strip)
        strip.save(Path(output_dir, output_filename))


def main(input_directories, image_strip_directory=None):

    output_directories = []

    for input_directory in input_directories:
        output_directory = np_dataset_to_png(input_directory)
        output_directories.append(output_directory)

    if image_strip_directory:
        if not image_strip_directory.is_dir():
            image_strip_directory.mkdir(parents=True, exist_ok=True)
        make_image_strips_multi_dir(output_directories, image_strip_directory)


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

    # _input_directories = [
    #     r'/home/acb6595/places/datasets/test/0003-tst-mp90_demo/rgb',
    #     r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/0-pan',
    #     r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/1-res',
    #     r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/2-blur',
    #     r'/home/acb6595/places/datasets/test/0002-tst-ep90_demo/3-noise',
    # ]

    # _input_directories = [
    #     r'/home/acb6595/sat6/datasets/test/0005-tst-mp90_demo/rgb',
    #     r'/home/acb6595/sat6/datasets/test/0005-tst-mp90_demo/0-pan',
    #     r'/home/acb6595/sat6/datasets/test/0005-tst-mp90_demo/1-res',
    #     r'/home/acb6595/sat6/datasets/test/0005-tst-mp90_demo/2-blur',
    #     r'/home/acb6595/sat6/datasets/test/0005-tst-mp90_demo/3-noise',
    # ]

    # _input_directories = [
    #     r'/home/acb6595/sat6/datasets/test/0007-tst-ep_demo/rgb',
    #     r'/home/acb6595/sat6/datasets/test/0007-tst-ep_demo/0-pan',
    #     r'/home/acb6595/sat6/datasets/test/0007-tst-ep_demo/1-res',
    #     r'/home/acb6595/sat6/datasets/test/0007-tst-ep_demo/2-blur',
    #     r'/home/acb6595/sat6/datasets/test/0007-tst-ep_demo/3-noise',
    # ]
    #
    # _input_directories = [
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/rgb',
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/0-pan',
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/1-res',
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/2-blur',
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/3-noise',
    # ]

    # _input_directories = [
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/rgb',
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/0-pan',
    #     r'/home/acb6595/places/datasets/demo/0000-demo-mp90_upsplash/3-noise',
    #     r'/home/acb6595/places/datasets/demo/0001-demo-ep90_upsplash/3-noise',
    # ]

    # _local_root = r'/home/acb6595/places/analysis/composite_performance/oct-models-fr90-mega-1-mega-2/3d/'
    # _input_directories = [
    #     r'exp_b0n0/predict_fit_slice_views/blur_slices/7-2',
    #     r'pl_b0n0/predict_fit_slice_views/blur_slices/7-2',
    #     r'giqe3_b2n2/predict_fit_slice_views/blur_slices/7-2',
    #     r'giqe5_b2n2/predict_fit_slice_views/blur_slices/7-2'
    # ]

    # _local_root = r'/home/acb6595/places/datasets/test/0006tst-pl-rgb'
    # _input_directories = [
    #     r'/home/acb6595/places/datasets/test/0006tst-pl-rgb/rgb',
    #     r'/home/acb6595/places/datasets/test/0006tst-pl-rgb/1-res',
    #     r'/home/acb6595/places/datasets/test/0006tst-pl-rgb/2-blur',
    #     r'/home/acb6595/places/datasets/test/0006tst-pl-rgb/3-noise'
    # ]

    _input_directories = [
        r'/home/acb6595/places/datasets/test/0007tst-pl-rgb-fr90-cp/rgb',
        r'/home/acb6595/places/datasets/test/0007tst-pl-rgb-fr90-cp/1-res',
        r'/home/acb6595/places/datasets/test/0007tst-pl-rgb-fr90-cp/2-blur',
        r'/home/acb6595/places/datasets/test/0007tst-pl-rgb-fr90-cp/3-noise'
    ]

    # _input_directories = [Path(_local_root, sub_dir) for sub_dir in _input_directories]

    # _output_directory_name = 'mp90_image_chain_rgb'
    # _output_directory_name = 'rgb_origin_mp90_ep90'
    # _output_directory_name = 'ep90_image_chain_rgb'
    # _output_directory_name = 'mp90_image_chain_rgb'
    # _output_directory_name = 'ep_image_chain_rgb'
    # _output_directory_name = 'upsplash_mp90_image_chain_rgb'
    # _output_directory_name = 'blur_slice_7-2_compare'
    _output_directory_name = 'pl-rgb-fr90-cp'

    _image_strip_output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['demo_images'], _output_directory_name)

    main(_input_directories, image_strip_directory=_image_strip_output_dir)

    # stack_dir = r'/home/acb6595/places/demo_images/upsplash_rgb_origin_mp90_ep90/keepers'
    # stack_dir = Path(stack_dir)
    # make_vertical_stack(stack_dir)

    # mosaic_test_dir = r'/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/rgb'
    # mosaic_test_output_dir = '/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/mosaic_test'
    # make_mosaic(mosaic_test_dir, output_dir=mosaic_test_output_dir, side_len=8)

    # mosaic_source_dirs = [
    #     r'/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/rgb',
    #     r'/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/0-pan',
    #     r'/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/1-res',
    #     r'/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/2-blur',
    #     r'/home/acb6595/sat6/demo_images/0005-tst-mp90_demo/3-noise',
    # ]
    # #
    # mosaic_source_dirs = [
    #     r'/home/acb6595/sat6/demo_images/0007-tst-ep_demo/rgb',
    #     r'/home/acb6595/sat6/demo_images/0007-tst-ep_demo/0-pan',
    #     r'/home/acb6595/sat6/demo_images/0007-tst-ep_demo/1-res',
    #     r'/home/acb6595/sat6/demo_images/0007-tst-ep_demo/2-blur',
    #     r'/home/acb6595/sat6/demo_images/0007-tst-ep_demo/3-noise',
    # ]
    #
    # # mosaic_output_dir = make_mosaics_multi_dir(mosaic_source_dirs)
    # # make_horizontal_strip(mosaic_output_dir)
    #
    # manual_img_strip_dir = r'/home/acb6595/sat6/demo_images/origin-mp-ep-rgb'
    # make_horizontal_strip(manual_img_strip_dir)

    # adding comment so that I can test a git commit on a remote system
