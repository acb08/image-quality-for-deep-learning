import copy

import matplotlib.pyplot as plt
import numpy as np
from src.d00_utils.functions import get_config
import argparse
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS
from src.d00_utils.functions import increment_suffix, log_config


def grouped_bar_chart(data, group_labels, ylabel='mean accuracy', group_width=0.7, padding=3, bar_width_frac=0.85,
                      edge_color='black', line_width=1, output_dir=None, x_scale=1,
                      figsize=(8, 8 / 1.33), label_threshold=None, manual_name=None, overwrite=False):

    x = np.arange(len(group_labels)) * x_scale
    num_items = len(data)
    bar_space = group_width / num_items
    bar_width = bar_width_frac * bar_space
    bar_offset = -1 * group_width / 2

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, item_data) in enumerate(data):
        left_edge = bar_offset + (i + 0.5) * bar_space
        rect = ax.bar(x + left_edge, item_data, bar_width, label=label, edgecolor=edge_color, linewidth=line_width)

        labels = [str(item)[1:] for item in item_data]  # strip off leading zeros (i.e. '0.01' -> '.01')
        labels = [f'{item}0' if len(item) == 2 else item for item in labels]
        if label_threshold:
            labels = [f'<{str(label_threshold)[1:]}' if float(item) < label_threshold else item for item in labels]
        ax.bar_label(rect, labels=labels, padding=padding)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('test dataset')
        ax.set_xticks(x, group_labels)
        ax.legend()

    fig.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir, 'bar_chart.png'))
    plt.show()


def get_output_dir(data, parent_dir='default', overwrite=True, suffix=None, manual_name=None):

    if parent_dir == 'default':
        parent_dir = Path(ROOT_DIR, REL_PATHS['bar_charts'])

    if not parent_dir.is_dir():
        Path.mkdir(parent_dir)

    if manual_name:
        new_dir_name = manual_name
    else:
        new_dir_name = None
        for item in data:
            label = item[0]
            result_num_string = label[:4]
            if not new_dir_name:
                new_dir_name = result_num_string
            else:
                new_dir_name = f'{new_dir_name}-{result_num_string}'

    if suffix:
        new_dir_name = f'{new_dir_name}-{suffix}'
    new_dir = Path(parent_dir, new_dir_name)

    if not new_dir.is_dir():
        Path.mkdir(new_dir)
        return new_dir
    elif overwrite:
        return new_dir
    else:
        if not suffix:
            suffix = 'v2'
            return get_output_dir(data, parent_dir=parent_dir, overwrite=overwrite, suffix=suffix,
                                  manual_name=manual_name)
        else:
            suffix = increment_suffix(suffix)
            return get_output_dir(data, parent_dir=parent_dir, overwrite=overwrite, suffix=suffix,
                                  manual_name=manual_name)


def main(run_config):

    output_dir = get_output_dir(run_config['data'], overwrite=run_config['overwrite'],
                                manual_name=run_config['manual_name'])
    run_config['output_dir'] = output_dir
    grouped_bar_chart(**run_config)
    log_config(output_dir, run_config)


if __name__ == '__main__':

    config_filename = 'bar_config.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'bar_chart'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    _run_config = get_config(args_passed)

    main(_run_config)
