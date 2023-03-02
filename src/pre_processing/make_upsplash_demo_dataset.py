import PIL.Image
import wandb
from pathlib import Path
from PIL import Image


def make_dataset_txt(directory, label=0, filename='upsplash_demo_original.txt', extensions=('jpg', 'png')):

    extensions = set(extensions)

    directory = Path(directory)
    image_filenames = list(directory.iterdir())
    image_filenames = [str(file_path.parts[-1]) for file_path in image_filenames]

    image_filenames = [image_filename for image_filename in image_filenames if image_filename[-3:] in extensions]

    with open(Path(directory, filename), 'w') as f:
        for image_filename in image_filenames:
            f.write(f'{image_filename} {label}\n')


def resize_256(directory, output_directory, names_labels_filename='upsplash_demo_original.txt'):

    new_size = (256, 256)
    if not Path(output_directory).is_dir():
        Path.mkdir(Path(output_directory), exist_ok=True, parents=True)

    with open(Path(directory, names_labels_filename), 'r') as f:

        for i, line in enumerate(f):

            filename_num_string = str(i).zfill(3)
            new_filename = f'demo256_{filename_num_string}.png'

            image_filename = line.split(' ')[0]
            image = Image.open(Path(directory, image_filename))
            image = image.resize(new_size, resample=PIL.Image.BICUBIC)
            image.save((Path(output_directory, new_filename)))


if __name__ == '__main__':

    _start_directory = r'/home/acb6595/places/datasets/demo/upsplash'
    _new_directory = r'/home/acb6595/places/datasets/demo/upsplash_256'
    resize_256(_start_directory, _new_directory)
    make_dataset_txt(_new_directory, filename='upsplash_demo_256.txt')
    # _files = make_dataset_txt(_directory)
    # print(_files)

