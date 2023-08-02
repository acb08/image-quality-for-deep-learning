import os

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage.measure import marching_cubes

from src.analysis.fit import linear_predict
from src.analysis.analysis_functions import conditional_extract_2d, create_identifier, get_distortion_perf_2d, \
    get_distortion_perf_1d, measure_log_perf_correlation, flatten, keep_2_of_3, sort_parallel, simple_model_check, \
    flatten_2x, keep_1_of_3, durbin_watson, get_2d_slices

# from src.analysis.distortion_performance import plot_perf_2d_multi_result, plot_perf_1d_multi_result

AZ_EL_DEFAULTS = {
    'az': -60,
    'el': 30
}

AZ_EL_COMBINATIONS_MINI = {
    '0-0': {'az': AZ_EL_DEFAULTS['az'], 'el': AZ_EL_DEFAULTS['el']},
    '7-2': {'az': 60, 'el': 20},
    '10-2': {'az': 105, 'el': 20},
}


# I do not remember why I structured AZ_EL_COMBINATIONS this way (strong suspicion the reason was stupid), but I do not
# think it's worth changing
AZ_EL_COMBINATIONS = {
    '0-0': {'az': AZ_EL_DEFAULTS['az'], 'el': AZ_EL_DEFAULTS['el']},
    '1-0': {'az': -30, 'el': 30},
    '2-0': {'az': -15, 'el': 30},
    '3-0': {'az': 0, 'el': 30},
    '4-0': {'az': 15, 'el': 30},
    '5-0': {'az': 30, 'el': 20},
    '6-0': {'az': 45, 'el': 30},
    '7-0': {'az': 60, 'el': 30},
    '8-0': {'az': 75, 'el': 30},
    '9-0': {'az': 90, 'el': 30},
    '10-0': {'az': 105, 'el': 30},
    '11-0': {'az': 120, 'el': 30},
    '12-0': {'az': 135, 'el': 30},

    '0-2': {'az': AZ_EL_DEFAULTS['az'], 'el': 20},
    '1-2': {'az': -30, 'el': 20},
    '2-2': {'az': -15, 'el': 20},
    '3-2': {'az': 0, 'el': 20},
    '4-2': {'az': 15, 'el': 20},
    '5-2': {'az': 30, 'el': 20},
    '6-2': {'az': 45, 'el': 20},
    '7-2': {'az': 60, 'el': 20},
    '8-2': {'az': 75, 'el': 20},
    '9-2': {'az': 90, 'el': 20},
    '10-2': {'az': 105, 'el': 20},
    '11-2': {'az': 120, 'el': 20},
    '12-2': {'az': 135, 'el': 20},


    '01': {'az': -60, 'el': 40},
    '11': {'az': -30, 'el': 40},
    '21': {'az': 30, 'el': 40},
    '31': {'az': 60, 'el': 40},
}

ISOSURF_AZ_EL_COMBINATIONS = {
    '0-0': {'az': AZ_EL_DEFAULTS['az'], 'el': AZ_EL_DEFAULTS['el']},
    '1-0': {'az': -30, 'el': 30},
    '5-0': {'az': 30, 'el': 20},
    '7-0': {'az': 60, 'el': 30},
    '11-0': {'az': 120, 'el': 30},

    '0-2': {'az': AZ_EL_DEFAULTS['az'], 'el': 20},
    '1-2': {'az': -30, 'el': 20},
    '5-2': {'az': 30, 'el': 20},
    '7-2': {'az': 60, 'el': 20},
    '11-2': {'az': 120, 'el': 20},

    '01': {'az': -60, 'el': 40},
    '11': {'az': -30, 'el': 40},
    '21': {'az': 30, 'el': 40},
    '31': {'az': 60, 'el': 40},
}


AZ_EL_META_DICT = {
    'all': AZ_EL_COMBINATIONS,
    'mini': AZ_EL_COMBINATIONS_MINI,
    'isosurf': ISOSURF_AZ_EL_COMBINATIONS
}

AXIS_LABELS = {
    'res': 'resolution',
    'blur': r'$\sigma$-blur',
    'scaled_blur': r'$\sigma$-blur scaled',
    'noise': r'$\sqrt{\lambda}$-noise',
    'z': 'accuracy',
    'y': 'accuracy',
    'acc': 'accuracy',
    'mpc': 'mean per class accuracy',
    'effective_entropy': 'effective entropy (bits)',
    'mAP': 'mAP'
}

SCATTER_PLOT_MARKERS = ['.', 'v', '2', 'P', 's', 'd', 'X', 'h']
COLORS = ['b', 'g', 'c', 'm', 'y', 'r', 'k', 'tab:orange']

AXIS_FONTSIZE = 14
LEGEND_FONTSIZE = 12


def plot_1d_linear_fit(x_data, y_data, fit_coefficients, distortion_id,
                       result_identifier=None, ylabel=None, title=None, directory=None,
                       per_class=False, show_plots=False):

    xlabel = AXIS_LABELS[distortion_id]
    x_plot = np.linspace(np.min(x_data), np.max(x_data), num=50)
    y_plot = fit_coefficients[0] * x_plot + fit_coefficients[1]

    ax = plt.figure().gca()

    ax.plot(x_plot, y_plot, linestyle='dashed', lw=0.8, color='k')
    ax.scatter(x_data, y_data)
    ax.set_xlabel(xlabel)
    if 'noise' in xlabel or np.max(x_data) > 5:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if not ylabel and not per_class:
        ylabel = AXIS_LABELS['y']
    elif not ylabel and per_class:
        ylabel = AXIS_LABELS['mpc']

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if directory:
        if result_identifier:
            save_name = f'{distortion_id}_{result_identifier}_{ylabel}'
        else:
            save_name = f'{distortion_id}_{ylabel}'
        save_name = save_name + '.png'
        plt.savefig(Path(directory, save_name))

    plt.tight_layout()

    if show_plots:
        plt.show()
    plt.close()


def plot_1d_fit(x, y_data, y_fit, distortion_id, measured_label='measured', fit_label='fit',
                result_identifier=None, ylabel=None, directory=None, legend=True, show_plots=True,
                ax=None, axis_fontsize=AXIS_FONTSIZE, legend_fontsize=LEGEND_FONTSIZE):

    xlabel = AXIS_LABELS[distortion_id]

    if ax is None:
        ax = plt.figure().gca()
        close_plot_here = False
    else:
        close_plot_here = True

    ax.plot(x, y_fit, label=fit_label, linestyle='dashed', lw=0.8, color='k')
    ax.scatter(x, y_data, color='k', marker='+', label=measured_label)
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    if 'noise' in xlabel or np.max(x) > 5:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if not ylabel:
        ylabel = AXIS_LABELS['y']
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.label_outer()

    if xlabel == 'resolution':
        legend_loc = 'lower right'
    elif xlabel[-4:] == 'blur':
        legend_loc = 'upper right'
    elif xlabel[-5:] == 'noise':
        legend_loc = 'upper right'
    else:
        legend_loc = 'best'

    if legend:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)

    if close_plot_here:
        plt.tight_layout()

    if directory:
        if result_identifier:
            save_name = f'{result_identifier}_{distortion_id}'
        else:
            save_name = f'{distortion_id}'
        save_name = save_name + '.png'
        plt.savefig(Path(directory, save_name))

    if show_plots:
        plt.show()

    if close_plot_here:
        plt.close()


def plot_2d(x_values, y_values, accuracy_means, x_id, y_id,
            zlabel=None,
            result_identifier=None,
            axis_labels=None,
            az_el_combinations='all',
            directory=None,
            show_plots=False,
            perf_metric='acc',
            z_limits=None,
            sub_dir_per_az_el=False):

    if not axis_labels or axis_labels == 'default':
        xlabel, ylabel, zlabel = AXIS_LABELS[x_id], AXIS_LABELS[y_id], AXIS_LABELS[perf_metric]
    elif axis_labels == 'effective_entropy_default':
        xlabel, ylabel, zlabel = AXIS_LABELS[x_id], AXIS_LABELS[y_id], AXIS_LABELS['effective_entropy']
    else:
        xlabel, ylabel = axis_labels[x_id], axis_labels[y_id]

    if result_identifier:
        save_name = f'{x_id}_{y_id}_{str(result_identifier)}_{perf_metric}.png'
    else:
        save_name = f'{x_id}_{y_id}_{perf_metric}.png'

    if az_el_combinations in AZ_EL_META_DICT.keys():

        combinations = AZ_EL_META_DICT[az_el_combinations]

        for combination_key in combinations.keys():

            if sub_dir_per_az_el:
                sub_dir = Path(directory, combination_key)
                if not sub_dir.is_dir():
                    Path.mkdir(sub_dir)
            else:
                sub_dir = directory

            az, el = combinations[combination_key]['az'], combinations[combination_key]['el']

            wire_plot(x_values, y_values, accuracy_means,
                      xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                      az=az, el=el,
                      save_name=save_name,
                      directory=sub_dir,
                      show_plots=show_plots,
                      z_limits=z_limits)

    else:
        if sub_dir_per_az_el:
            raise Warning('sub_dir_per_az_el only works when az_el_combinations in AZ_EL_META_DICT.keys()')
        if az_el_combinations == 'default':
            az, el = AZ_EL_DEFAULTS['az'], AZ_EL_DEFAULTS['el']
        elif az_el_combinations in AZ_EL_COMBINATIONS:
            az, el = AZ_EL_COMBINATIONS[az_el_combinations]['az'], AZ_EL_COMBINATIONS[az_el_combinations]['el']
        else:
            az, el = az_el_combinations[0], az_el_combinations[1]

        wire_plot(x_values, y_values, accuracy_means,
                  xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                  az=az, el=el,
                  save_name=save_name,
                  directory=directory,
                  show_plots=show_plots,
                  z_limits=z_limits)


def wire_plot(x, y, z,
              xlabel='x',
              ylabel='y',
              zlabel='default',
              title=None,
              directory=None,
              save_name=None,
              az=AZ_EL_DEFAULTS['az'],
              el=AZ_EL_DEFAULTS['el'],
              alpha=0.5,
              indexing='ij',
              show_plots=False,
              z_limits=None):

    fig = plt.figure()
    ax = plt.axes(projection='3d', azim=az, elev=el)

    if isinstance(z, dict):
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, key in enumerate(z):

            if isinstance(x, dict):
                assert isinstance(y, dict)
                xx, yy = np.meshgrid(x[key], y[key], indexing=indexing)
            else:
                xx, yy = np.meshgrid(x, y, indexing=indexing)

            if isinstance(alpha, list):
                alpha_plot = alpha[i]
            else:
                alpha_plot = alpha

            ax.plot_wireframe(xx, yy, z[key], label=str(key), color=color_list[i], alpha=alpha_plot)
        ax.legend()

    else:
        xx, yy = np.meshgrid(x, y, indexing=indexing)
        ax.plot_wireframe(xx, yy, z, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if z_limits is not None:
        ax.set_zlim(z_limits[0], z_limits[1])
    if zlabel:
        if zlabel == 'default':
            zlabel = AXIS_LABELS['z']
        ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save_name:
        if az != AZ_EL_DEFAULTS['az'] or el != AZ_EL_DEFAULTS['el']:
            seed = save_name.split('.')[0]
            az_int = int(az)
            el_int = int(el)
            save_name = f"{seed}_az{az_int}_el{el_int}.png"
    if directory and save_name:
        plt.savefig(Path(directory, save_name))

    if show_plots:
        fig.show()
    plt.close(fig)


def plot_2d_linear_fit(distortion_array, accuracy_means, fit, x_id, y_id,
                       result_identifier=None, axis_labels='default', az_el_combinations='all', directory=None,
                       show_plots=False):
    x, y = distortion_array[:, 0], distortion_array[:, 1]
    predict_mean_accuracy_vector = linear_predict(fit, distortion_array)
    x_values, y_values, predicted_mean_accuracy, __ = conditional_extract_2d(x, y, predict_mean_accuracy_vector)

    z_plot = {
        'fit': predicted_mean_accuracy,
        'actual': accuracy_means
    }

    plot_2d(x_values, y_values, z_plot, x_id, y_id,
            axis_labels=axis_labels, az_el_combinations=az_el_combinations, directory=directory,
            result_identifier=result_identifier, show_plots=show_plots)


def analyze_plot_results_together(model_results, identifier=None, directory=None, make_subdir=False, dim_tag='2d',
                                  legend_loc='best', pairwise_analysis=False, log_file=None, create_log_file=False,
                                  show_plots=False, single_legend=True):
    """
    Plots performance of multiple models together. If pairwise_analysis is True and len(model_results) is 2, measures
    the correlation coefficients between mean performances as a function of distortion combinations.
    """
    if not identifier:
        if dim_tag == '2d':
            identifier = create_identifier(model_results)
        else:
            identifier = create_identifier(model_results, dim_tag=dim_tag)

    if directory and make_subdir:
        sub_dir = Path(directory, identifier)
        if not sub_dir.is_dir():
            Path.mkdir(sub_dir, parents=True)

        directory = sub_dir

    if not log_file and create_log_file:
        if pairwise_analysis:
            log_file = open(Path(directory, 'pairwise_result_log.txt'), 'w')
        else:
            log_file = open(Path(directory, 'auto_result_log.txt'), 'w')

    if dim_tag == '2d':
        analyze_plot_perf_2d_multi_result(model_results, directory=directory, identifier=identifier,
                                          pairwise_analysis=pairwise_analysis, log_file=log_file)
    else:
        analyze_plot_perf_1d_multi_result(model_results, directory=directory, identifier=identifier,
                                          legend_loc=legend_loc,
                                          pairwise_analysis=pairwise_analysis, log_file=log_file,
                                          show_plots=show_plots,
                                          single_legend=single_legend)
    if log_file is not None:
        log_file.close()


def analyze_plot_perf_1d_multi_result(model_performances,
                                      distortion_ids=('res', 'blur', 'noise'),
                                      directory=None,
                                      identifier=None,
                                      legend_loc='best',
                                      pairwise_analysis=False,
                                      log_file=None,
                                      show_plots=False,
                                      plot_together=True,
                                      single_legend=True,
                                      perform_fits=True):
    """
    :param model_performances: list of model performance class instances
    :param distortion_ids: distortion type tags to be analyzed
    :param directory: output directory
    :param identifier: str for use as a filename seed
    :param legend_loc: str to specify plot legend location
    :param pairwise_analysis: bool, correlations between performance results measured if True
    :param log_file: text fle for logging analysis results
    :param show_plots: bool, determines whether plots are displayed
    :param plot_together: bool, determines whether to use subplots sharing y-axis
    :param single_legend: bool, determines whether all subplots have a legend or only right-most subplot
    :param perform_fits: bool, determines whether linear fits generated when get_distortion_perf_1d() is called
    """
    if plot_together:
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 3.4))
        save_dir_individual = None
        show_plots_individual = False
        if single_legend:
            legends = len(distortion_ids) * [False]
            legends[-1] = True
        else:
            legends = len(distortion_ids) * [True]

    else:
        axes = None, None, None
        fig = None
        save_dir_individual = directory
        show_plots_individual = show_plots
        legends = len(distortion_ids) * [True]

    for i, distortion_id in enumerate(distortion_ids):
        ax = axes[i]
        mean_performances = {}
        for model_performance in model_performances:
            performance_key = str(model_performance)
            x, y, fit_coefficients, fit_correlation = get_distortion_perf_1d(model_performance, distortion_id,
                                                                             perform_fit=perform_fits)
            mean_performances[performance_key] = y

        if pairwise_analysis:
            if len(model_performances) != 2:
                raise Exception('model_performances must be length 2 for pairwise_analysis')

            measure_log_perf_correlation(mean_performances, distortion_ids=distortion_id, log_file=log_file)

        plot_1d_performance(x, mean_performances, distortion_id, result_identifier=identifier,
                            directory=save_dir_individual,
                            legend_loc=legend_loc, show_plots=show_plots_individual,
                            ax=ax, legend=legends[i])

    if plot_together:
        fig.tight_layout()

        if directory:
            save_name = identifier
            for distortion_id in distortion_ids:
                save_name = f'{save_name}_{distortion_id}'
            save_name = save_name + '.png'
            fig.savefig(Path(directory, save_name))

    if show_plots:
        plt.show()


def analyze_plot_perf_2d_multi_result(model_performances,
                                      distortion_ids=('res', 'blur', 'noise'),
                                      distortion_combinations=((0, 1), (1, 2), (0, 2)),
                                      directory=None,
                                      add_bias=True,
                                      identifier=None,
                                      pairwise_analysis=False,
                                      log_file=None,
                                      show_plots=False):

    for i, (idx_0, idx_1) in enumerate(distortion_combinations):

        x_id, y_id = distortion_ids[idx_0], distortion_ids[idx_1]
        mean_performances = {}

        for model_performance in model_performances:
            performance_key = str(model_performance)
            x_values, y_values, accuracy_means, fit, corr, distortion_arr = get_distortion_perf_2d(model_performance,
                                                                                                   x_id,
                                                                                                   y_id,
                                                                                                   add_bias=add_bias,
                                                                                                   log_file=log_file)
            mean_performances[performance_key] = accuracy_means

        if pairwise_analysis:
            if len(model_performances) != 2:
                raise Exception('model_performances must be length 2 for pairwise_analysis')

            measure_log_perf_correlation(mean_performances, distortion_ids=(x_id, y_id), log_file=log_file)

        plot_2d(x_values, y_values, mean_performances, x_id, y_id,
                result_identifier=identifier,
                axis_labels='default',
                az_el_combinations='all',
                directory=directory,
                show_plots=show_plots)


def conditional_multi_plot_3d(blur_sigmas, noise_means, z_dict,
                              xlabel=r'$\sigma$ Gaussian blur',
                              ylabel=r'$\lambda$ Poisson noise',
                              zlabel='default',
                              title=None,
                              folder=None,
                              save_name=None,
                              az=AZ_EL_DEFAULTS['az'],
                              el=AZ_EL_DEFAULTS['el'],
                              indexing='ij',
                              lw=1):
    z_extract_dict = {}

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for key in z_dict:
        sigma_vals, mean_vals, z_extract_dict[key], __ = (
            conditional_extract_2d(blur_sigmas,
                                   noise_means,
                                   z_dict[key]))

    X, Y = np.meshgrid(sigma_vals, mean_vals, indexing=indexing)
    fig = plt.figure()
    ax = plt.axes(projection='3d', azim=az, elev=el)
    counter = 0
    for key in z_extract_dict:
        ax.plot_wireframe(X, Y, z_extract_dict[key], label=str(key),
                          color=color_list[counter], linewidth=lw)
        counter += 1
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if zlabel:
        if zlabel == 'default':
            zlabel = AXIS_LABELS['z']
        ax.set_zlabel(zlabel)
    ax.set_title(title)
    # ax.view_init(ax, el)
    fig.tight_layout()

    if save_name:
        if az != AZ_EL_DEFAULTS['az'] or el != AZ_EL_DEFAULTS['el']:
            seed = save_name.split('.')[0]
            azInt = int(az)
            elInt = int(el)
            save_name = f"{seed}_az{azInt}_el{elInt}.png"

    if folder and save_name:
        plt.savefig(os.path.join(folder, save_name))

    fig.show()

    return save_name


# def plot_isosurf(vol_data, x, y, z, scx=1, scy=1, scz=1,
#                  levels=None, save_name=None, save_dir=None,
#                  x_label='resolution', y_label='blur', z_label='noise',
#                  alpha=0.2, step_size=1, az_el_combinations='default',
#                  show_plots=False):
#     """
#     Uses the marching cubes method to identify iso-surfaces in 3d data and then creates a 3d plot of the iso-surface at
#     the value specified by level. If level==None, the mean of the min and max value in vol_data is used.
#     """
#
#     delta_x = x[1] - x[0]
#     delta_y = y[1] - y[0]
#     delta_z = z[1] - z[0]
#     x0, y0, z0 = np.min(x), np.min(y), np.min(z)
#
#     vertices_list = []
#     faces_list = []
#
#     if not levels:
#         try:
#             verts, faces, normals, values = marching_cubes(vol_data,
#                                                            spacing=(delta_x, delta_y, delta_z),
#                                                            step_size=step_size)
#             # slide vertices by appropriate offset in x, y, z since marching cubes does not account
#             # for absolute coordinates
#             offset = np.multiply([x0, y0, z0], np.ones(np.shape(verts)))
#             verts += offset
#
#             vertices_list.append(verts)
#             faces_list.append(faces)
#         except ValueError:
#             print('unable to generate iso-surface')
#
#     else:
#         for level in levels:
#             mean, std = np.mean(vol_data), np.std(vol_data)
#             scaled_level = level * std + mean
#             try:
#                 verts, faces, normals, values = marching_cubes(vol_data,
#                                                                spacing=(delta_x, delta_y, delta_z),
#                                                                level=scaled_level,
#                                                                step_size=step_size)
#                 # slide vertices by appropriate offset in x, y, z since marching cubes does not account
#                 # for absolute coordinates
#                 offset = np.multiply([x0, y0, z0], np.ones(np.shape(verts)))
#                 verts += offset
#
#                 vertices_list.append(verts)
#                 faces_list.append(faces)
#             except ValueError:
#                 print(f'unable to generate iso-surface at level {level}')
#
#     if az_el_combinations == 'all':
#         az_el_combinations_local = AZ_EL_COMBINATIONS
#     elif az_el_combinations == 'default':
#         az_el_combinations_local = {'0-0': {'az': AZ_EL_DEFAULTS['az'], 'el': AZ_EL_DEFAULTS['el']}}
#     elif az_el_combinations == 'iso_default':
#         az_el_combinations_local = ISOSURF_AZ_EL_COMBINATIONS
#     else:
#         raise ValueError('az_el_combinations must be either "all", "default", or "iso_default"')
#
#     for combination_key in az_el_combinations_local:
#         az, el = az_el_combinations_local[combination_key]['az'], az_el_combinations_local[combination_key]['el']
#
#         fig = plt.figure(figsize=(10, 10))
#         ax = fig.add_subplot(111, projection='3d', azim=az, elev=el)
#
#         for i, verts in enumerate(vertices_list):
#             faces = faces_list[i]
#             mesh = Poly3DCollection(verts[faces], alpha=alpha)
#             mesh.set_edgecolor('k')
#             ax.add_collection3d(mesh)
#
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         ax.set_zlabel(z_label)
#
#         ax.set_xlim(min(0, scx * np.min(x)), scx * np.max(x))
#         ax.set_ylim(min(0, scy * np.min(y)), scy * np.max(y))
#         ax.set_zlim(min(0, scz * np.min(z)), scz * np.max(z))
#         plt.tight_layout()
#
#         if az_el_combinations != 'default':
#             save_name_stem = save_name.split('.')[0]
#             save_name_updated = f'{save_name_stem}_az{az}_el{el}.png'
#         else:
#             save_name_updated = save_name
#
#         if save_name and save_dir:
#             plt.savefig(Path(save_dir, save_name_updated))
#
#         if show_plots:
#             plt.show()
#         plt.close()


def plot_1d_performance(x, performance_dict, distortion_id,
                        result_identifier=None,
                        xlabel='default',
                        ylabel='default',
                        directory=None,
                        legend_loc='best',
                        legend=True,
                        show_plots=False,
                        ax=None,
                        axis_fontsize=AXIS_FONTSIZE,
                        legend_fontsize=LEGEND_FONTSIZE,
                        close_plot_here=False,
                        y_lim_bottom=None,
                        y_lim_top=None):

    if xlabel == 'default':
        xlabel = AXIS_LABELS[distortion_id]
    if ylabel == 'default':
        ylabel = AXIS_LABELS['y']

    if result_identifier:
        save_name = f'{distortion_id}_{str(result_identifier)}_acc'
    else:
        save_name = f'{distortion_id}_acc'

    if legend_loc and legend_loc != 'best':
        save_name = f"{save_name}_{legend_loc.replace(' ', '_')}"

    save_name = f"{save_name}.png"

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        if xlabel == 'resolution':
            legend_loc = 'lower right'
        elif xlabel[-4:] == 'blur':
            legend_loc = 'upper right'
        elif xlabel[-5:] == 'noise':
            legend_loc = 'upper right'
        else:
            legend_loc = 'best'

    for i, key in enumerate(performance_dict):
        if type(x) == dict:
            x_plot = x[key]
        else:
            x_plot = x
        ax.plot(x_plot, performance_dict[key], color=COLORS[i])
        ax.scatter(x_plot, performance_dict[key], label=key, c=COLORS[i], marker=SCATTER_PLOT_MARKERS[i])
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    if y_lim_bottom is not None or y_lim_top is not None:
        ax.set_ylim(bottom=y_lim_bottom, top=y_lim_top)
    ax.label_outer()

    if legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.tight_layout()

    if directory:
        plt.savefig(Path(directory, save_name))
    if show_plots:
        plt.show()

    if close_plot_here:
        plt.close()


def compare_2d_slice_views(f0, f1, x_vals, y_vals, z_vals, distortion_ids=('res', 'blur', 'noise'),
                           slice_axes=(0, 1, 2), slices_interval=1, data_labels=('f0', 'f1'), result_id='2d_slice',
                           az_el_combinations='mini', directory=None, show_plots=False,
                           perf_metric='acc', sub_dir_per_az_el=True):

    z_min = 0.9 * min(np.min(f0), np.min(f1))
    z_max = 1.1 * max(np.max(f0), np.max(f1))
    z_limits = (z_min, z_max)

    if f1 is None:
        assert type(f0) == dict

    if f1 is None:
        raise NotImplementedError

    for slice_axis in slice_axes:

        xlabel, ylabel = keep_2_of_3(a=distortion_ids, discard_idx=slice_axis)
        slice_axis_label = distortion_ids[slice_axis]

        slice_axis_sub_dir = Path(directory, f'{slice_axis_label}_slices')
        if not slice_axis_sub_dir.is_dir():
            Path.mkdir(slice_axis_sub_dir)

        if f1 is not None:

            f0_slices, axis0, axis1 = get_2d_slices(f0, x_vals, y_vals, z_vals,
                                                    slice_axis=slice_axis,
                                                    slice_interval=slices_interval)
            f1_slices, __, __ = get_2d_slices(f1, x_vals, y_vals, z_vals,
                                              slice_axis=slice_axis,
                                              slice_interval=slices_interval)

            assert f0_slices.keys() == f1_slices.keys()

            for i, (key, f0_slice) in enumerate(f0_slices.items()):
                views_2d = {
                    data_labels[0]: f0_slice,
                    data_labels[1]: f1_slices[key]
                }

                plot_2d(axis0, axis1, views_2d, x_id=xlabel, y_id=ylabel, result_identifier=f'{result_id}_{i}',
                        az_el_combinations=az_el_combinations, directory=slice_axis_sub_dir, show_plots=show_plots,
                        perf_metric=perf_metric, z_limits=z_limits, sub_dir_per_az_el=sub_dir_per_az_el)

        else:
            raise NotImplementedError


def compare_2d_mean_views(f0, f1, x_vals, y_vals, z_vals, distortion_ids=('res', 'blur', 'noise'),
                          flatten_axes=(0, 1, 2), data_labels=('f0', 'f1'), result_id='3d_projection',
                          az_el_combinations='default', directory=None, residual_plot=False, show_plots=False,
                          perf_metric='acc'):

    if f1 is None:
        assert type(f0) == dict

    for flatten_axis in flatten_axes:

        xlabel, ylabel = keep_2_of_3(a=distortion_ids, discard_idx=flatten_axis)
        axis0, axis1 = None, None

        if f1 is not None:
            f0_2d, axis0, axis1 = flatten(f0, x_vals, y_vals, z_vals, flatten_axis=flatten_axis)
            f1_2d, __, __ = flatten(f1, x_vals, y_vals, z_vals, flatten_axis=flatten_axis)
            # assert np.array_equal(axis0, _axis0)
            # assert np.array_equal(axis1, _axis1)

            views_2d = {
                data_labels[0]: f0_2d,
                data_labels[1]: f1_2d
            }

            if residual_plot:
                residual = f0_2d - f1_2d
                views_2d['residual'] = residual
                result_id = f'{result_id}_rdl'
        else:

            views_2d = {}

            for key, data3d in f0.items():

                f_2d, _axis0, _axis1 = flatten(data3d, x_vals, y_vals, z_vals, flatten_axis=flatten_axis)

                if axis0 is not None and axis1 is not None:
                    assert np.array_equal(axis0, _axis0)
                    assert np.array_equal(axis1, _axis1)
                axis0, axis1 = _axis0, _axis1

                views_2d[key] = f_2d

        plot_2d(axis0, axis1, views_2d, x_id=xlabel, y_id=ylabel, result_identifier=result_id,
                az_el_combinations=az_el_combinations, directory=directory, show_plots=show_plots,
                perf_metric=perf_metric)


def compare_1d_views(f0, f1, x_vals, y_vals, z_vals, distortion_ids=('res', 'blur', 'noise'),
                     flatten_axis_combinations=((1, 2), (0, 2), (0, 1)), data_labels=('measured', 'fit'),
                     result_id='3d_1d_projection', directory=None, include_fit_stats=True, include_dw=False,
                     show_plots=True, plot_together=True, ylabel=None):

    if plot_together:
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 3.4))
        save_dir_individual = None
        show_plots_individual = False
    else:
        axes = None, None, None
        fig = None
        save_dir_individual = directory
        show_plots_individual = show_plots

    for i, flatten_axes in enumerate(flatten_axis_combinations):
        f0_1d, axis = flatten_2x(f0, x_vals, y_vals, z_vals, flatten_axes=flatten_axes)
        f1_1d, __ = flatten_2x(f1, x_vals, y_vals, z_vals, flatten_axes=flatten_axes)
        axis_label = keep_1_of_3(a=distortion_ids, discard_indices=flatten_axes)
        axis_check = keep_1_of_3(a=x_vals, b=y_vals, c=z_vals, discard_indices=flatten_axes)
        assert np.array_equal(axis, axis_check)

        if include_fit_stats:
            rho = np.corrcoef(f0_1d, f1_1d)[0, 1]
            rho = round(rho, 3)
            corr_str = r'$\rho = $'
            if include_dw:
                dw = durbin_watson(prediction=f0_1d, measurement=f1_1d)
                dw = round(dw, 3)
                fit_label = f'{data_labels[1]}, {corr_str} {rho}, Durbin-Watson = {dw}'
            else:
                fit_label = f'{data_labels[1]}, {corr_str} {rho}'
        else:
            fit_label = data_labels[1]

        plot_1d_fit(axis, f0_1d, f1_1d, axis_label, measured_label=data_labels[0], fit_label=fit_label,
                    result_identifier=result_id, directory=save_dir_individual, show_plots=show_plots_individual,
                    ax=axes[i], ylabel=ylabel)

    if plot_together:
        fig.tight_layout()

        if directory:
            save_name = result_id
            for distortion_id in distortion_ids:
                save_name = f'{save_name}_{distortion_id}'
            save_name = save_name + '.png'
            fig.savefig(Path(directory, save_name))

    if show_plots:
        plt.show()


def plot_1d_from_3d(perf_dict_3d, x_vals, y_vals, z_vals, distortion_ids=('res', 'blur', 'noise'),
                    result_identifier='3d_1d_projection',
                    flatten_axis_combinations=((1, 2), (0, 2), (0, 1)),
                    directory=None,
                    show_plots=True, plot_together=True,
                    name_string=None,
                    ylabel='mAP',
                    legend=False,
                    y_lim_bottom=None,
                    y_lim_top=None,
                    single_legend=True,
                    subfig_width=None,
                    subfig_height=None):

    if subfig_width is None:
        subfig_width = 4
    if subfig_height is None:
        subfig_height = 3.4

    num_combinations = len(flatten_axis_combinations)

    if plot_together:
        fig_width = subfig_width * num_combinations
        fig, axes = plt.subplots(nrows=1, ncols=num_combinations, sharey=True, figsize=(fig_width, subfig_height))
        if not hasattr(axes, '__len__'):
            axes = (axes, )
        save_dir_individual = None
        show_plots_individual = False
        if single_legend:
            legends = num_combinations * [False]
            legends[-1] = True
        else:
            legends = num_combinations * [True]

    else:
        axes = num_combinations * [None]
        fig = None
        save_dir_individual = directory
        show_plots_individual = show_plots
        legends = len(distortion_ids) * [True]

    performance_dict_1d = {}

    for i, flatten_axes in enumerate(flatten_axis_combinations):
        axis = None
        axis_label = None
        # prev_axis = None
        for key, perf_3d in perf_dict_3d.items():

            # if type(x_vals) == dict:
            #     assert type(y_vals) == dict
            #     assert type(z_vals) == dict
            #     x_plot = x_vals[key]
            #     y_plot = y_vals[key]
            #     z_plot = z_vals[key]
            # else:
            #     x_plot = x_vals
            #     y_plot = y_vals
            #     z_plot = z_vals

            f0_1d, axis = flatten_2x(perf_3d, x_vals, y_vals, z_vals, flatten_axes=flatten_axes)
            axis_label = keep_1_of_3(a=distortion_ids, discard_indices=flatten_axes)
            # axis_check = keep_1_of_3(a=x_vals, b=y_vals, c=z_vals, discard_indices=flatten_axes)
            # assert np.array_equal(axis, axis_check)
            # if prev_axis is not None:
            #     assert np.array_equal(axis, prev_axis)
            # prev_axis = axis

            performance_dict_1d[key] = f0_1d

        plot_1d_performance(x=axis,
                            performance_dict=performance_dict_1d,
                            distortion_id=axis_label,
                            result_identifier=result_identifier,
                            ylabel=ylabel,
                            directory=save_dir_individual,
                            show_plots=show_plots_individual,
                            ax=axes[i],
                            legend=legends[i],
                            y_lim_bottom=y_lim_bottom,
                            y_lim_top=y_lim_top,
                            )

    if plot_together:
        fig.tight_layout()

        if directory:
            save_name = result_identifier
            for distortion_id in distortion_ids:
                save_name = f'{save_name}_{distortion_id}'
            save_name = save_name + '.png'
            fig.savefig(Path(directory, save_name))

    if show_plots:
        plt.show()


def plot_1d(x, y, directory=None, filename=None, xlabel='x', ylabel='y', literal_xlabel=False, literal_ylabel=False,
            ax=None, show=True):

    if xlabel in AXIS_LABELS.keys() and not literal_xlabel:
        xlabel = AXIS_LABELS[xlabel]
    if ylabel in AXIS_LABELS.keys() and not literal_ylabel:
        ylabel = AXIS_LABELS[ylabel]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if directory is not None and filename is not None:
        plt.savefig(Path(directory, filename))

    if show:
        plt.show()


def residual_color_plot(f0, f1, x_vals, y_vals, z_vals, distortion_ids=('res', 'blur', 'noise'),
                        flatten_axes=(0, 1, 2), directory=None):

    residual = f0 - f1
    fig, axes = plt.subplots(nrows=1, ncols=len(flatten_axes))
    residual_arrays = []
    xy_axes = []

    xy_labels = []

    for i, flatten_axis in enumerate(flatten_axes):
        r2d, axis0, axis1 = flatten(residual, x_vals, y_vals, z_vals, flatten_axis=flatten_axis)
        assert np.shape(r2d) == (len(axis0), len(axis1))
        xlabel, ylabel = keep_2_of_3(a=distortion_ids, discard_idx=flatten_axis)
        # xlabel = AXIS_LABELS[xlabel]
        # ylabel = AXIS_LABELS[ylabel]

        residual_arrays.append(r2d)
        xy_axes.append((axis0, axis1))
        xy_labels.append((xlabel, ylabel))

    vmin = np.min([np.min(r2d) for r2d in residual_arrays])
    vmax = np.max([np.max(r2d) for r2d in residual_arrays])

    for i, r2d in enumerate(residual_arrays):
        ax = axes[i]
        x_axis, y_axis = xy_axes[i]
        xlabel, ylabel = xy_labels[i]
        img = ax.imshow(r2d, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.38, 0.05, 0.3])
    fig.colorbar(img, cax=cbar_ax)


def _heat_plot(arr, xlabel, ylabel, ax=None, vmin=None, vmax=None, extent=None):

    # plt.figure()
    # plt.imshow(arr, vmin=vmin, vmax=vmax, extent=extent)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.colorbar()

    if not ax:
        fig, ax = plt.subplots()

    ax.imshow(arr, extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def sorted_linear_scatter(prediction, result, directory=None, filename='predict_result_scatter.png', best_fit=True,
                          xlabel='predicted accuracy', ylabel='accuracy', show_plots=True, ax=None,
                          legend_fontsize=LEGEND_FONTSIZE, axis_fontsize=AXIS_FONTSIZE):

    prediction, result = sort_parallel(prediction, result)

    x_min, x_max = np.min(prediction), np.max(prediction)
    x = np.linspace(x_min, x_max)

    if best_fit:
        slope, intercept, r_squared = simple_model_check(prediction, result, pre_sorted=True)
        y = slope * x + intercept
        label = fr'$y = {round(slope, 3)} x + {round(intercept, 3)}$, $r^2={round(r_squared, 3)}$'

    else:
        y = x
        label = '1-to-1'

    if ax is None:
        plt.figure()
        plt.scatter(prediction, result, marker=".", s=0.5)
        plt.plot(x, y, linestyle='--', color='k', label=label)
        plt.xlabel(xlabel, fontsize=axis_fontsize)
        plt.ylabel(ylabel, fontsize=axis_fontsize)
        plt.legend(fontsize=legend_fontsize)
        if directory:
            plt.savefig(Path(directory, filename))
        plt.tight_layout()
        if show_plots:
            plt.show()
        plt.close()

    else:
        ax.scatter(prediction, result, marker=".", s=0.5)
        ax.plot(x, y, linestyle='--', color='k', label=label)
        ax.set_xlabel(xlabel, fontsize=axis_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        ax.legend(loc='upper left', fontsize=legend_fontsize)
        ax.label_outer()


def dual_sorted_linear_scatter(prediction_0, result_0, prediction_1, result_1, directory=None,
                               filename='predict_result_scatter_combined.png', best_fit=True,
                               xlabel_0='predicted accuracy', xlabel_1='predicted accuracy',
                               ylabel='accuracy', show_plots=True,
                               legend_fontsize=LEGEND_FONTSIZE, axis_fontsize=AXIS_FONTSIZE):

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8), sharey=True)

    sorted_linear_scatter(prediction_0, result_0, directory=None, filename=None, best_fit=best_fit,
                          xlabel=xlabel_0, ylabel=ylabel, show_plots=False, ax=ax0,
                          legend_fontsize=legend_fontsize, axis_fontsize=axis_fontsize)
    sorted_linear_scatter(prediction_1, result_1, directory=None, filename=None, best_fit=best_fit,
                          xlabel=xlabel_1, ylabel=ylabel, show_plots=False, ax=ax1,
                          legend_fontsize=legend_fontsize, axis_fontsize=axis_fontsize)
    fig.tight_layout()

    if directory:
        plt.savefig(Path(directory, filename))

    if show_plots:
        fig.show()

