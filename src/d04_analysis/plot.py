import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

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
    'noise': r'$\sqrt{\lambda}$-noise',
    'z': 'accuracy',
    'y': 'accuracy',
    'effective_entropy': 'effective entropy (bits)'
}

SCATTER_PLOT_MARKERS = ['.', 'v', '2', 'P', 's', 'd', 'X', 'h']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot_1d_linear_fit(x_data, y_data, fit_coefficients, distortion_id,
                       result_identifier=None, ylabel='accuracy', title=None, directory=None):
    xlabel = AXIS_LABELS[distortion_id]
    x_plot = np.linspace(np.min(x_data), np.max(x_data), num=50)
    y_plot = fit_coefficients[0] * x_plot + fit_coefficients[1]

    ax = plt.figure().gca()

    ax.plot(x_plot, y_plot, linestyle='dashed', lw=0.8, color='k')
    ax.scatter(x_data, y_data)
    ax.set_xlabel(xlabel)
    if 'noise' in xlabel or np.max(x_data) > 5:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if directory:
        if result_identifier:
            save_name = f'{distortion_id}_{result_identifier}_{ylabel}.png'
        else:
            save_name = f'{distortion_id}_{ylabel}.png'
        plt.savefig(Path(directory, save_name))
    plt.show()


def plot_2d(x_values, y_values, accuracy_means, x_id, y_id,
            result_identifier=None,
            axis_labels=None,
            az_el_combinations='all',
            directory=None):
    if not axis_labels or axis_labels == 'default':
        xlabel, ylabel = AXIS_LABELS[x_id], AXIS_LABELS[y_id]
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
                      xlabel=xlabel, ylabel=ylabel,
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