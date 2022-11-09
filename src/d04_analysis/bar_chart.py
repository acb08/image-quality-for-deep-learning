import copy
from src.d04_analysis.analysis_functions import consolidate_fit_stats
import matplotlib.pyplot as plt
import numpy as np
from src.d00_utils.functions import get_config
import argparse
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS
from src.d00_utils.functions import increment_suffix, log_config


def grouped_bar_chart(data, group_labels, ylabel='mean accuracy', xlabel=None, group_width=0.7, padding=3, bar_width_frac=0.85,
                      edge_color='black', line_width=1, output_dir=None, x_scale=1,
                      figsize=(8, 8 / 1.33), label_threshold=None, include_bar_labels=True, rotation=45,
                      include_legend=True):

    x = np.arange(len(group_labels)) * x_scale
    num_items = len(data)
    bar_space = group_width / num_items
    bar_width = bar_width_frac * bar_space
    bar_offset = -1 * group_width / 2

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, item_data) in enumerate(data):
        left_edge = bar_offset + (i + 0.5) * bar_space
        rect = ax.bar(x + left_edge, item_data, bar_width, label=label, edgecolor=edge_color, linewidth=line_width)

        if include_bar_labels:
            labels = [round(item, 2) for item in item_data]
            labels = [str(item)[1:] for item in labels]  # strip off leading zeros (i.e. '0.01' -> '.01')
            labels = [f'{item}0' if len(item) == 2 else item for item in labels]
            if label_threshold:
                labels = [f'<{str(label_threshold)[1:]}' if float(item) < label_threshold else item for item in labels]
            ax.bar_label(rect, labels=labels, padding=padding)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x, group_labels, rotation=rotation)

    if include_legend:
        ax.legend(loc='upper right')

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


def sort_for_bar_chart(consolidated_stats, target_keys=('res', 'blur', 'noise'), traverse_keys=('1d',),
                       outer_keys=None):

    if outer_keys is None:
        outer_keys = list(consolidated_stats.keys())

    sorted_data = []

    if target_keys == 'all':
        sub_dict = consolidated_stats[outer_keys[0]]
        if traverse_keys:
            for traverse_key in traverse_keys:
                sub_dict = sub_dict[traverse_key]
            target_keys = list(sub_dict.keys())

    for target_key in target_keys:
        target_data = []
        for outer_key in outer_keys:
            sub_dict = consolidated_stats[outer_key]
            if traverse_keys:
                for traverse_key in traverse_keys:
                    sub_dict = sub_dict[traverse_key]

            target = sub_dict[target_key]
            target_data.append(target)

        sorted_data.append([
            target_key, target_data
        ])

    return outer_keys, sorted_data


def main(run_config):

    """
    Generates a grouped bar chart using either data in run_config or else data fit characterization data extracted and
    grouped with the consolidate_fit_stats() and sort_for_bar_chart() functions.
    """
    if 'data' not in run_config.keys():

        fit_keys = run_config['fit_keys']
        composite_result_id = run_config['composite_result_id']
        analysis_type = run_config['analysis_type']
        target_keys = run_config['target_keys']
        traverse_keys = run_config['traverse_keys']

        group_labels = run_config['group_labels']

        consolidated_fit_stats = consolidate_fit_stats(fit_keys=fit_keys,
                                                       composite_result_id=composite_result_id,
                                                       analysis_type=analysis_type)
        __, data = sort_for_bar_chart(consolidated_fit_stats, target_keys=target_keys,
                                      traverse_keys=traverse_keys, outer_keys=fit_keys)
        if group_labels is None:
            group_labels = copy.deepcopy(fit_keys)

    else:
        data = run_config['data']
        group_labels = run_config['group_labels']

    overwrite = run_config['overwrite']
    manual_name = run_config['manual_name']
    ylabel = run_config['ylabel']
    xlabel = run_config['xlabel']
    group_width = run_config['group_width']
    padding = run_config['padding']
    bar_width_frac = run_config['bar_width_frac']
    edge_color = run_config['edge_color']
    line_width = run_config['line_width']
    label_threshold = run_config['label_threshold']
    include_bar_labels = run_config['include_bar_labels']
    rotation = run_config['rotation']
    include_legend = run_config['include_legend']

    output_dir = get_output_dir(data, overwrite=overwrite,
                                manual_name=manual_name)

    grouped_bar_chart(data=data,
                      group_labels=group_labels,
                      ylabel=ylabel,
                      xlabel=xlabel,
                      group_width=group_width,
                      padding=padding,
                      bar_width_frac=bar_width_frac,
                      edge_color=edge_color,
                      line_width=line_width,
                      output_dir=output_dir,
                      label_threshold=label_threshold,
                      include_bar_labels=include_bar_labels,
                      rotation=rotation,
                      include_legend=include_legend)

    log_config(output_dir, run_config)


if __name__ == '__main__':

    config_filename = 'giqe3_places_3d.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'bar_chart'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    _run_config = get_config(args_passed)

    main(_run_config)
