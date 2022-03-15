import numpy as np
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.d04_analysis.plot_defaults import AZ_EL_DEFAULTS, AXIS_LABELS
from pathlib import Path
import os


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


def build_3d_field(x, y, z, f, data_dump=False):
    """
    x: array of length N containing nx unique values
    y: array of length N containing ny unique values
    z: array of length N containing nz unique values
    f: array of length N, with values that will be sorted into a 3D (nx, ny, nz)-shape
    array, with indices (i, j, k), where 0 <= i <= nx-1, 0 <= j <= ny-1, 0 <= k <= nz-1,
    where each (i, j, k) represents a unique combination
    of the unique values of x, y, and z. In other words, lets imagine that f is a
    3d function of two variables sigma and lambda, and that we have N samples
    of z, with each sample corresponding to a pair values sigma and lambda.
    This function extracts the relevant values of z for each unique
    (sigma, lambda) pair.

    returns:

        x_values: numpy array, where x_values[alpha] represents the alpha-th unique
        value of x
        y_values: numpy array, where y_values[beta] represents the beta-th unique
        value of y
        z_means: j x k array, where z_means[alpha, beta] is the mean of z
        where x == x_values[alpha] and y == y_values[beta]
        extracts: dictionary, where keys are tuples (alpha, beta) and values are
        1D numpy arrays of z values where x == alpha and y == beta

    """

    full_extract = {}  # diagnostic
    x_values = np.unique(x)
    y_values = np.unique(y)
    z_values = np.unique(z)
    f_means = np.zeros((len(x_values), len(y_values), len(z_values)))

    parameter_array = []  # for use in curve fits
    performance_array = []  # for use in svd

    for i, x_val in enumerate(x_values):
        x_inds = np.where(x == x_val)
        for j, y_val in enumerate(y_values):
            y_inds = np.where(y == y_val)
            for k, z_val in enumerate(z_values):
                z_inds = np.where(z == z_val)
                xy_inds = np.intersect1d(x_inds, y_inds)
                xyz_inds = np.intersect1d(xy_inds, z_inds)

                full_extract[(x_val, y_val, z_val)] = f[xyz_inds]
                f_means[i, j, k] = np.mean(f[xyz_inds])
                parameter_array.append([x_val, y_val, z_val])
                performance_array.append(f_means[i, j, k])

    if data_dump:
        parameter_array = np.asarray(parameter_array, dtype=np.float32)
        performance_array = np.atleast_2d(np.asarray(performance_array, dtype=np.float32)).T
        return x_values, y_values, z_values, f_means, parameter_array, performance_array, full_extract
    else:
        return f_means


def conditional_extract_2d(x, y, z):
    """
    x: array of length N containing j unique values
    y: array of length N containing k unique values
    z: array of length N, with values that will be sorted into a 2D j-by-k
    array, with indices (alpha, beta), where 0 <= alpha <= j-1, 0 <= beta <= k-1,
    where each (alpha, beta) represents a unique combination
    of the unique values of x and y. In other words, lets imagine that z is a
    2d function of two variables sigma and lambda, and that we have N samples
    of z, with each sample corresponding to a pair values sigma and lambda.
    This function extracts the relevant values of z for each unique
    (sigma, lambda) pair.

    returns:

        x_values: numpy array, where x_values[alpha] represents the alpha-th unique
        value of x
        y_values: numpy array, where y_values[beta] represents the beta-th unique
        value of y
        z_means: j x k array, where z_means[alpha, beta] is the mean of z
        where x == x_values[alpha] and y == y_values[beta]
        extracts: dictionary, where keys are tuples (alpha, beta) and values are
        1D numpy arrays of z values where x == alpha and y == beta

    """

    x_values = np.unique(x)
    y_values = np.unique(y)
    z_means = np.zeros((len(x_values), len(y_values)))

    param_array = []  # for use in curve fits
    performance_array = []  # for use in svd

    for x_counter, x_val in enumerate(x_values):
        x_inds = np.where(x == x_val)
        for y_counter, y_val in enumerate(y_values):
            y_inds = np.where(y == y_val)
            z_inds = np.intersect1d(x_inds, y_inds)
            z_means[x_counter, y_counter] = np.mean(z[z_inds])
            param_array.append([x_val, y_val])
            performance_array.append(z_means[x_counter, y_counter])

    # full extract arrays written out this way for use in svd.
    vector_data_extract = {
        'param_array': np.asarray(param_array),
        'performance_array': np.atleast_2d(performance_array).T
    }

    return x_values, y_values, z_means, vector_data_extract


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


if __name__ == '__main__':
    _x = np.random.randint(0, 10, 10000)
    _y = np.random.randint(0, 10, 10000)
    _z = np.random.randint(0, 10, 10000)
    # _f = np.sqrt(_x**2 + _y++2 + _z**2)
    # _f = np.sqrt(_x**2 + _y + _z)
    _f = _x ** 2 + _y ** 2 + _z ** 2
    _x_vals, _y_vals, _z_vals, _f_means, _full_extract = build_3d_field(_x, _y, _z, _f)
    plot_isosurf(_f_means, _x_vals, _y_vals, _z_vals, level=25)
