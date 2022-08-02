import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from src.d04_analysis.fit import linear_predict
from src.d04_analysis.analysis_functions import conditional_extract_2d, create_identifier, get_distortion_perf_2d, \
    get_distortion_perf_1d, measure_log_perf_correlation, flatten, keep_2_of_3, sort_parallel, simple_model_check

# from src.d04_analysis.distortion_performance import plot_perf_2d_multi_result, plot_perf_1d_multi_result

AZ_EL_DEFAULTS = {
    'az': -60,
    'el': 30
}

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

AXIS_LABELS = {
    'res': 'resolution',
    'blur': r'$\sigma$-blur',
    'scaled_blur': r'$\sigma$-blur scaled',
    'noise': r'$\sqrt{\lambda}$-noise',
    'z': 'accuracy',
    'y': 'accuracy',
    'mpc': 'mean per class accuracy',
    'effective_entropy': 'effective entropy (bits)'
}

SCATTER_PLOT_MARKERS = ['.', 'v', '2', 'P', 's', 'd', 'X', 'h']
COLORS = ['b', 'g', 'c', 'm', 'y', 'r', 'k']


def plot_1d_linear_fit(x_data, y_data, fit_coefficients, distortion_id,
                       result_identifier=None, ylabel=None, title=None, directory=None,
                       per_class=False):
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
    plt.show()


def plot_2d(x_values, y_values, accuracy_means, x_id, y_id,
            zlabel=None,
            result_identifier=None,
            axis_labels=None,
            az_el_combinations='all',
            directory=None):
    if not axis_labels or axis_labels == 'default':
        xlabel, ylabel, zlabel = AXIS_LABELS[x_id], AXIS_LABELS[y_id], AXIS_LABELS['z']
    elif axis_labels == 'effective_entropy_default':
        xlabel, ylabel, zlabel = AXIS_LABELS[x_id], AXIS_LABELS[y_id], AXIS_LABELS['effective_entropy']
    else:
        xlabel, ylabel = axis_labels[x_id], axis_labels[x_id]

    if result_identifier:
        save_name = f'{x_id}_{y_id}_{str(result_identifier)}_acc.png'
    else:
        save_name = f'{x_id}_{y_id}_acc.png'

    if az_el_combinations == 'all':

        for combination_key in AZ_EL_COMBINATIONS:
            az, el = AZ_EL_COMBINATIONS[combination_key]['az'], AZ_EL_COMBINATIONS[combination_key]['el']

            wire_plot(x_values, y_values, accuracy_means,
                      xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                      az=az, el=el,
                      save_name=save_name,
                      directory=directory)

    else:
        if az_el_combinations == 'default':
            az, el = AZ_EL_DEFAULTS['az'], AZ_EL_DEFAULTS['el']
        elif az_el_combinations in AZ_EL_COMBINATIONS:
            az, el = AZ_EL_COMBINATIONS[az_el_combinations]['az'], AZ_EL_COMBINATIONS[az_el_combinations]['el']
        else:
            az, el = az_el_combinations[0], az_el_combinations[1]

        wire_plot(x_values, y_values, accuracy_means,
                  xlabel=xlabel, ylabel=ylabel,
                  az=az, el=el,
                  save_name=save_name,
                  directory=directory)


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
              indexing='ij'):

    xx, yy = np.meshgrid(x, y, indexing=indexing)
    fig = plt.figure()
    ax = plt.axes(projection='3d', azim=az, elev=el)

    if isinstance(z, dict):
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, key in enumerate(z):
            if isinstance(alpha, list):
                alpha_plot = alpha[i]
            else:
                alpha_plot = alpha
            ax.plot_wireframe(xx, yy, z[key], label=str(key), color=color_list[i], alpha=alpha_plot)
        ax.legend()
    else:
        ax.plot_wireframe(xx, yy, z, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if zlabel:
        if zlabel == 'default':
            zlabel = AXIS_LABELS['z']
        ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    if save_name:
        if az != AZ_EL_DEFAULTS['az'] or el != AZ_EL_DEFAULTS['el']:
            seed = save_name.split('.')[0]
            az_int = int(az)
            el_int = int(el)
            save_name = f"{seed}_az{az_int}_el{el_int}.png"
    if directory and save_name:
        plt.savefig(Path(directory, save_name))
    fig.show()


def plot_2d_linear_fit(distortion_array, accuracy_means, fit, x_id, y_id,
                       result_identifier=None, axis_labels='default', az_el_combinations='all', directory=None):
    x, y = distortion_array[:, 0], distortion_array[:, 1]
    predict_mean_accuracy_vector = linear_predict(fit, distortion_array)
    x_values, y_values, predicted_mean_accuracy, __ = conditional_extract_2d(x, y, predict_mean_accuracy_vector)

    z_plot = {
        'fit': predicted_mean_accuracy,
        'actual': accuracy_means
    }

    plot_2d(x_values, y_values, z_plot, x_id, y_id,
            axis_labels=axis_labels, az_el_combinations=az_el_combinations, directory=directory,
            result_identifier=result_identifier)


def analyze_plot_results_together(model_results, identifier=None, directory=None, make_subdir=False, dim_tag='2d',
                                  legend_loc='best', pairwise_analysis=False, log_file=None, create_log_file=False):
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
                                          pairwise_analysis=pairwise_analysis, log_file=log_file)
    if log_file is not None:
        log_file.close()


def analyze_plot_perf_1d_multi_result(model_performances,
                                      distortion_ids=('res', 'blur', 'noise'),
                                      directory=None,
                                      identifier=None,
                                      legend_loc='best',
                                      pairwise_analysis=False,
                                      log_file=None):
    """
    :param model_performances: list of model performance class instances
    :param distortion_ids: distortion type tags to be analyzed
    :param directory: output derectory
    :param identifier: str for use as a filename seed
    :param legend_loc: str to specify plot legend location
    :param pairwise_analysis: bool, correlations between performance results measured if True
    :param log_file: text fle for logging analysis results
    """

    for i, distortion_id in enumerate(distortion_ids):
        mean_performances = {}
        for model_performance in model_performances:
            performance_key = str(model_performance)
            x, y, fit_coefficients, fit_correlation = get_distortion_perf_1d(model_performance, distortion_id)
            mean_performances[performance_key] = y

        if pairwise_analysis:
            if len(model_performances) != 2:
                raise Exception('model_performances must be length 2 for pairwise_analysis')

            measure_log_perf_correlation(mean_performances, distortion_ids=distortion_id, log_file=log_file)

        plot_1d_performance(x, mean_performances, distortion_id, result_identifier=identifier, directory=directory,
                            legend_loc=legend_loc)


def analyze_plot_perf_2d_multi_result(model_performances,
                                      distortion_ids=('res', 'blur', 'noise'),
                                      distortion_combinations=((0, 1), (1, 2), (0, 2)),
                                      directory=None,
                                      add_bias=True,
                                      identifier=None,
                                      pairwise_analysis=False,
                                      log_file=None):

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
                directory=directory)


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


def conditional_plot_3d(blur_sigmas, noise_means, z,
                        xlabel=r'$\sigma$ Gaussian blur',
                        ylabel=r'$\lambda$ Poisson noise',
                        zlabel='default',
                        title=None,
                        folder=None,
                        save_name=None,
                        az=AZ_EL_DEFAULTS['az'],
                        el=AZ_EL_DEFAULTS['el'],
                        indexing='ij'):

    sigma_vals, mean_vals, blur_noise_Shannon_entropy_2d, __ = (
        conditional_extract_2d(blur_sigmas,
                               noise_means,
                               z))

    X, Y = np.meshgrid(sigma_vals, mean_vals, indexing=indexing)
    fig = plt.figure()
    ax = plt.axes(projection='3d', azim=az, elev=el)
    ax.plot_wireframe(X, Y, blur_noise_Shannon_entropy_2d)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if zlabel:
        if zlabel == 'default':
            zlabel = AXIS_LABELS['z']
        ax.set_zlabel(zlabel)
    ax.set_title(title)

    # ax.view_init(ax, el)

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


def plot_isosurf(vol_data, x, y, z, scx=1, scy=1, scz=1,
                 level=None, save_name=None, save_dir=None,
                 x_label='resolution', y_label='blur', z_label='noise',
                 az=30, el=30, alpha=0.2, step_size=1, verbose=False):
    """
    Uses the marching cubes method to identify iso-surfaces in 3d data and then creates a 3d plot of the iso-surface at
    the value specified by level. If level==None, the mean of the min and max value in vol_data is used.
    """

    delta_x = x[1] - x[0]
    delta_y = y[1] - y[0]
    delta_z = z[1] - z[0]

    if not level:
        verts, faces, normals, values = marching_cubes(vol_data,
                                                       spacing=(delta_x, delta_y, delta_z),
                                                       step_size=step_size)
    else:
        verts, faces, normals, values = marching_cubes(vol_data,
                                                       spacing=(delta_x, delta_y, delta_z),
                                                       level=level,
                                                       step_size=step_size)

    # slide vertices by appropriate offset in x, y, z since marching cubes does not account
    # for absolute coordinates
    x0, y0, z0 = np.min(x), np.min(y), np.min(z)
    offset = np.multiply([x0, y0, z0], np.ones(np.shape(verts)))
    verts += offset

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d', azim=az, elev=el)

    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.set_xlim(min(0, scx * np.min(x)), scx * np.max(x))
    ax.set_ylim(min(0, scy * np.min(y)), scy * np.max(y))
    ax.set_zlim(min(0, scz * np.min(z)), scz * np.max(z))
    plt.tight_layout()
    if save_name and save_dir:
        plt.savefig(Path(save_dir, save_name))
    plt.show()

    if verbose:
        return verts, faces, normals, values


def plot_1d_performance(x, performance_dict, distortion_id,
                        result_identifier=None,
                        xlabel='default',
                        ylabel='default',
                        directory=None,
                        legend_loc='best',
                        legend=True):

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

    plt.figure()
    for i, key in enumerate(performance_dict):
        plt.plot(x, performance_dict[key], color=COLORS[i])
        plt.scatter(x, performance_dict[key], label=key, c=COLORS[i], marker=SCATTER_PLOT_MARKERS[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(loc=legend_loc)
    if directory:
        plt.savefig(Path(directory, save_name))
    plt.show()


def compare_2d_views(f0, f1, x_vals, y_vals, z_vals, distortion_ids=('res', 'blur', 'noise'),
                     flatten_axes=(0, 1, 2), data_labels=('f0', 'f1'), result_id='3d_projection',
                     az_el_combinations='default', directory=None, residual_plot=False):

    for flatten_axis in flatten_axes:
        f0_2d, axis0, axis1 = flatten(f0, x_vals, y_vals, z_vals, flatten_axis=flatten_axis)
        f1_2d, __, __ = flatten(f1, x_vals, y_vals, z_vals, flatten_axis=flatten_axis)
        xlabel, ylabel = keep_2_of_3(a=distortion_ids, discard_idx=flatten_axis)

        views_2d = {
            data_labels[0]: f0_2d,
            data_labels[1]: f1_2d
        }

        if residual_plot:
            residual = f0_2d - f1_2d
            views_2d['residual'] = residual
            result_id = f'{result_id}_rdl'

        plot_2d(axis0, axis1, views_2d, x_id=xlabel, y_id=ylabel, result_identifier=result_id,
                az_el_combinations=az_el_combinations, directory=directory)


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

    fig.show()


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
                          xlabel='predicted accuracy', ylabel='accuracy'):

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

    plt.figure()
    plt.scatter(prediction, result, marker=".", s=0.5)
    plt.plot(x, y, linestyle='--', color='k', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if directory:
        plt.savefig(Path(directory, filename))
    plt.show()


if __name__ == '__main__':

    _res_vals = np.linspace(0.1, 1, num=10)
    _blur_vals = np.linspace(0.1, 5, num=20)
    _noise_vals = np.arange(40) * 2

    _f0 = np.random.rand(len(_res_vals), len(_blur_vals), len(_noise_vals))
    _f1 = np.random.rand(len(_res_vals), len(_blur_vals), len(_noise_vals))

    residual_color_plot(_f0, _f1, _res_vals, _blur_vals, _noise_vals)
