"""
Concatenates equivalent plots from different directories into strips for easier comparison.
"""

from src.pre_processing import demo_view
from pathlib import Path
import argparse
from src.utils import definitions, functions


def make_plot_strip(config):

    parent_dir = config['parent_dir']
    input_sub_dirs = config['input_sub_dirs']
    output_dir = config['output_dir']
    base_directory_idx = config['base_directory_idx']

    parent_dir = Path(definitions.ROOT_DIR, parent_dir)

    input_dirs = [Path(parent_dir, sub_dir) for sub_dir in input_sub_dirs]
    output_dir = Path(definitions.ROOT_DIR, parent_dir, output_dir)
    if not output_dir.is_dir():
        Path.mkdir(output_dir, exist_ok=True)

    demo_view.make_image_strips_multi_dir(input_dirs, output_dir, base_directory_idx=base_directory_idx)

    functions.log_config(output_dir, config)


if __name__ == '__main__':

    config_filename = 'coco128-original-mp90-ep90.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'plot_compare_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = functions.get_config(args_passed)

    make_plot_strip(run_config)
