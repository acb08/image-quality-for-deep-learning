import copy
from src.d04_analysis.analysis_functions import consolidate_fit_stats
import matplotlib.pyplot as plt
import numpy as np
from src.d00_utils.functions import get_config
import argparse
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, REL_PATHS
from src.d00_utils.functions import increment_suffix, log_config
from src.d04_analysis.fit_functions import generate_fit_keys

BAR_HATCHES = [
    '////',
    '+++',
    'xxxx',
    '...',
    '***',
    'ooo'
]


BAR_COLORS = [
    'tab:blue',
    'tab:cyan',
    'tab:olive',
    'tab:orange',
    'tab:gray',
    'tab:green',
    'tab:purple',
    'tab:pink',
    'tab:red',
    'tab:brown',
]

FONT_SIZE = 11


def grouped_bar_chart(data, group_labels, ylabel='mean accuracy', xlabel=None, group_width=0.7, padding=3,
                      bar_width_frac=0.85,
                      edge_color='black', line_width=1, output_dir=None, x_scale=1,
                      figsize=(8, 8 / 1.33), label_threshold=None, include_bar_labels=True, rotation=45,
                      include_legend=True, bar_hatching=True):

    x = np.arange(len(group_labels)) * x_scale
    num_items = len(data)
    bar_space = group_width / num_items
    bar_width = bar_width_frac * bar_space
    bar_offset = -1 * group_width / 2

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, item_data) in enumerate(data):

        if bar_hatching and len(data) <= len(BAR_HATCHES):
            hatch = BAR_HATCHES[i]
        else:
            hatch = None

        if len(data) <= len(BAR_COLORS):
            color = BAR_COLORS[i]
        else:
            color = None

        left_edge = bar_offset + (i + 0.5) * bar_space
        rect = ax.bar(x + left_edge, item_data, bar_width, label=label, edgecolor=edge_color, linewidth=line_width,
                      hatch=hatch, color=color)

        if include_bar_labels:
            labels = [round(item, 2) for item in item_data]
            labels = [str(item)[1:] for item in labels]  # strip off leading zeros (i.e. '0.01' -> '.01')
            labels = [f'{item}0' if len(item) == 2 else item for item in labels]
            if label_threshold:
                labels = [f'<{str(label_threshold)[1:]}' if float(item) < label_threshold else item for item in labels]
            ax.bar_label(rect, labels=labels, padding=padding)

    ax.set_ylabel(ylabel, size=FONT_SIZE)
    ax.set_xlabel(xlabel, size=FONT_SIZE)
    ax.set_xticks(x, group_labels, rotation=rotation, size=FONT_SIZE)

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


def sort_filter_fit_stats_for_grouped_bar_chart(consolidated_stats, target_keys=('res', 'blur', 'noise'),
                                                traverse_keys=('1d',), outer_keys=None, additional_filter=None):
    """
    Sorts through nested dictionary structure to extract data to be used by grouped_bar_chart() function. The
    grouped_bar_chart() function takes data with the structure
    [
    data_label_0, --> i.e. the label that goes into the legend of the bar chart
        [
        group_0_value,
        group_1_value,
        ...
        ]
    data_label_1,
        [
        group_0_value,
        group_1_value,
        ...
        ]
    ],
    and it is into this structure that this function places data from consolidated_stats.


    :param consolidated_stats: dictionary containing nested dictionaries of fit statistics. Structured as follows:
    {outer_key: --> the fit_key used mapped to a particular fit function
        {
        traverse_key_0:  --> if present, key mapped to method of fit stat collection (e.g. dw in 1d vs. 2d)
            {
            target_key_0: target_value_0,
            target_key_1: target_value_1,
            ...
            }
        traverse_key_1:
            {
            target_key_0: target_value_0
            target_key_1: target_value_0
            ...
            }
        target_key_0:  target_value_0--> if travers_keys is None, target_keys can be found at first level of nested dict
        ...
        }
    }
    :param target_keys: keys corresponding to the actual data to be extracted
    :param traverse_keys:  intermediate keys mapped to method of stat collection
    :param outer_keys: keys to separate the separate fit functions
    :param additional_filter:  Not implemented yet
    :return:
        outer_keys --> list of the fit function keys used
        target_data --> list structured as follows:
        [target_key,
            [target_value_0,
            target_value_1,
            ...
            ]
        ]
    """

    if outer_keys is None:
        outer_keys = list(consolidated_stats.keys())

    if additional_filter:
        filter_func = sub_dict_filter_functions[additional_filter]
        sorted_data = apply_filter_func(consolidated_stats,
                                        outer_keys=outer_keys,
                                        traverse_keys=traverse_keys,
                                        filter_func=filter_func)
        return outer_keys, sorted_data

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


def apply_filter_func(consolidated_stats, outer_keys, traverse_keys, filter_func):

    sorted_data = []

    # note: traverse_keys used differently than in sort_filter_fit_stats_for_grouped_bar_chart()
    for i, traverse_key in enumerate(traverse_keys):
        target_data = []
        composite_key = None
        for outer_key in outer_keys:
            sub_dict = consolidated_stats[outer_key]

            sub_sub_dict = sub_dict[traverse_key]

            target_key_stand_in, target_key, target_val = filter_func(sub_sub_dict)
            if not composite_key:
                composite_key = f'{traverse_key}_{target_key_stand_in}'

            target_data.append(target_val)

        sorted_data.append(
            [composite_key, target_data]
        )

    return sorted_data


def min_value(data):

    keys = tuple(data.keys())
    values = tuple(data.values())
    min_val = min(values)
    min_val_idx = values.index(min_val)
    min_val_key = keys[min_val_idx]

    return 'min', min_val_key, min_val


def mean(data):

    # keys = tuple(data.keys())
    values = tuple(data.values())
    mean_val = np.mean(values)
    # min_val_idx = values.index(min_val)
    # min_val_key = keys[min_val_idx]

    return 'mean', None, mean_val


sub_dict_filter_functions = {
    'min_value': min_value,
    'mean': mean,
}


def main(run_config):

    """
    Generates a grouped bar chart using either data in run_config or else data fit characterization data extracted and
    grouped with the consolidate_fit_stats() and sort_filter_fit_stats_for_grouped_bar_chart() functions.
    """
    if 'data' not in run_config.keys():

        if 'fit_keys' in run_config.keys():
            fit_keys = run_config['fit_keys']
        else:
            functional_forms = run_config['functional_forms']
            blur_mappings = run_config['blur_mappings']
            noise_mappings = run_config['noise_mappings']
            fit_keys = generate_fit_keys(functional_forms, blur_mappings, noise_mappings)

        composite_result_id = run_config['composite_result_id']
        analysis_type = run_config['analysis_type']
        target_keys = run_config['target_keys']
        traverse_keys = run_config['traverse_keys']

        group_labels = run_config['group_labels']

        if 'additional_filter' in run_config.keys():
            additional_filter = run_config['additional_filter']
        else:
            additional_filter = None

        consolidated_fit_stats = consolidate_fit_stats(fit_keys=fit_keys,
                                                       composite_result_id=composite_result_id,
                                                       analysis_type=analysis_type)
        __, data = sort_filter_fit_stats_for_grouped_bar_chart(consolidated_fit_stats, target_keys=target_keys,
                                                               traverse_keys=traverse_keys, outer_keys=fit_keys,
                                                               additional_filter=additional_filter)
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
    if 'bar_hatching' in run_config.keys():
        bar_hatching = run_config['bar_hatching']
    else:
        bar_hatching = False

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
                      include_legend=include_legend,
                      bar_hatching=bar_hatching
                      )

    log_config(output_dir, run_config)


if __name__ == '__main__':

    config_filename = 'places_summary_threshold.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'bar_chart'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    _run_config = get_config(args_passed)

    main(_run_config)
