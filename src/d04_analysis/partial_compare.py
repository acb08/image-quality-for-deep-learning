import numpy as np
import matplotlib.pyplot as plt
from src.d00_utils.functions import load_npz_data
from pathlib import Path
from src.d00_utils.definitions import REL_PATHS, ROOT_DIR, STANDARD_PERFORMANCE_PREDICTION_FILENAME

class Partials(object):

    def __init__(self, x, y, z, partial_func_x=None, partial_func_y=None, partial_func_z=None,
                 xlabel='x', ylabel='y', zlabel='z', func_3d_eval=None, rescale_numeric_derivatives=True):

        # coordinates
        self.x = x
        self.y = y
        self.z = z

        self.rescale_numeric_derivatives = rescale_numeric_derivatives

        self._xx, self._yy, self._zz = np.meshgrid(x, y, z)

        self.func_3d_eval = func_3d_eval

        # partial derivative functions (functions of (x, y, z) in general)
        self._partial_func_x = partial_func_x
        self._partial_func_y = partial_func_y
        self._partial_func_z = partial_func_z

        self.f_x, self.f_y, self.f_z = self.evaluate_partials()
        self.gradient_magnitude = self._grad_mag()
        self.gradient = (self.f_x, self.f_y, self.f_z)
        self.coords = (self.x, self.y, self.z)

        self.shape = np.shape(self.gradient_magnitude)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.axis_labels = (self.xlabel, self.ylabel, self.zlabel)

    def evaluate_partials(self):

        if self.func_3d_eval is None:

            f_x_eval = self._partial_func_x(self._xx, self._yy, self._zz)
            f_y_eval = self._partial_func_y(self._xx, self._yy, self._zz)
            f_z_eval = self._partial_func_z(self._xx, self._yy, self._zz)

            return f_x_eval, f_y_eval, f_z_eval

        else:  # evaluate partial derivatives numerically
            return self.evaluate_partials_numeric()

    def _grad_mag(self):
        return np.sqrt(self.f_x**2 + self.f_y**2 + self.f_z**2)

    def isolate_plane_indices(self, axis_label, value, approx_ok=True):

        axis_num = self.axis_labels.index(axis_label)
        if value is None:
            return axis_num, None

        axis_values = self.coords[axis_num]
        if approx_ok:
            idx = np.argmin(np.abs(axis_values - value))
        else:
            idx = np.where(axis_values == value)[0][0]

        return axis_num, idx

    def evaluate_partials_numeric(self):

        dx = np.mean(np.diff(self.x))
        dy = np.mean(np.diff(self.y))
        dz = np.mean(np.diff(self.z))

        if self.func_3d_eval is None:
            raise ValueError

        f_x_eval = np.diff(self.func_3d_eval, axis=0)
        f_x_eval = f_x_eval[:, 1:, 1:]
        f_x_eval = f_x_eval / dx

        f_y_eval = np.diff(self.func_3d_eval, axis=1)
        f_y_eval = f_y_eval[1:, :, 1:]
        f_y_eval = f_y_eval / dy

        f_z_eval = np.diff(self.func_3d_eval, axis=2)
        f_z_eval = f_z_eval[1:, 1:, :]
        f_z_eval = f_z_eval / dz

        self._prune_coords()
        self._xx, self._yy, self._zz = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        return f_x_eval, f_y_eval, f_z_eval

    def _prune_coords(self):
        self.x = self.x[1:]
        self.y = self.y[1:]
        self.z = self.z[1:]

    def half_step_coord_offset(self):
        self.x = self.x + np.mean(np.diff(self.x)) / 2
        self.y = self.y + np.mean(np.diff(self.y)) / 2
        self.z = self.z + np.mean(np.diff(self.z)) / 2

    def coords_3d(self):
        return self._xx, self._yy, self._zz

    @staticmethod
    def extract_2d_slice(array, axis_num, idx):
        if axis_num == 0:
            return array[idx, :, :]
        elif axis_num == 1:
            return array[:, idx, :]
        elif axis_num == 2:
            return array[:, :, idx]
        else:
            raise ValueError('axis_num must be either 0, 1, or 2')

    def partial_mag_ratio(self, partial_deriv_axis_label, plane_definition_axis_label, plane_definition_val,
                          approx_ok=True):

        partial_deriv_axis_num, __ = self.isolate_plane_indices(axis_label=partial_deriv_axis_label, value=None)
        plane_def_axis_num, plane_def_idx = self.isolate_plane_indices(axis_label=plane_definition_axis_label,
                                                                       value=plane_definition_val,
                                                                       approx_ok=approx_ok)

        partial_deriv = self.gradient[partial_deriv_axis_num]
        partial_deriv = self.extract_2d_slice(partial_deriv, plane_def_axis_num, plane_def_idx)
        gradient_mag = self.extract_2d_slice(self.gradient_magnitude, plane_def_axis_num, plane_def_idx)

        return np.abs(partial_deriv) / gradient_mag, partial_deriv, gradient_mag


def grad_plot(xx, yy, zz, u, v, w,
              length=0.1, normalize=True):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(xx, yy, zz, u, v, w,
              normalize=normalize, length=length)
    fig.show()


def load_3d_perf_prediction(composite_result_id, analysis_type='3d', fit_key='giqe3_b3n2'):
    parent_dir = Path(ROOT_DIR, REL_PATHS['composite_performance'], composite_result_id, analysis_type, fit_key,
                      REL_PATHS['perf_prediction'])
    data = load_npz_data(parent_dir, STANDARD_PERFORMANCE_PREDICTION_FILENAME)
    perf_prediction_3d = data['perf_prediction_3d']
    x = data['x']
    y = data['y']
    z = data['z']
    distortion_ids = data['distortion_ids']

    return perf_prediction_3d, x, y, z, distortion_ids

GIQE_COEFFICIENTS = (
    9.57,  # A0
    -3.32,  # A1
    3.32,  # A2
    -1.9,  # A3
    -2,  # A4
    -1.8  # A5
)


def get_niirs(gsd, rer, snr):

    """

    :param gsd: ground sample distance (inches)
    :param rer:
    :param snr:
    :return:
    """

    a0 = GIQE_COEFFICIENTS[0]
    a1 = GIQE_COEFFICIENTS[1]
    a2 = GIQE_COEFFICIENTS[2]
    a3 = GIQE_COEFFICIENTS[3]
    a4 = GIQE_COEFFICIENTS[4]
    a5 = GIQE_COEFFICIENTS[5]
    pass


def p_x(xx, yy, zz):
    return 2 * xx


def p_y(xx, yy, zz):
    return 2 * yy


def p_z(xx, yy, zz):
    return 2 * zz


if __name__ == '__main__':

    _num = 16
    _x, _y, _z = np.linspace(0, 1, num=_num), np.linspace(0, 1, num=_num), np.linspace(0, 1, num=_num)

    pc1 = Partials(_x, _y, _z,
                   partial_func_x=p_x,
                   partial_func_y=p_y,
                   partial_func_z=p_z)

    _xx, _yy, _zz = np.meshgrid(_x, _y, _z)
    _f = _xx**2 + _yy**2 + _zz**2
    pc2 = Partials(_x, _y, _z,
                   func_3d_eval=_f)

    gm1 = pc1.gradient_magnitude
    gm2 = pc2.gradient_magnitude

    diff = gm1[1:, 1:, 1:] - gm2
    print(np.mean(diff), np.std(diff))
    # _u, _v, _w = pc1.gradient
    # grad_plot(_xx, _yy, _zz, _u, _v, _w)

    # result_id= 'oct-models-fr90-mega-1-mega-2'
    result_id = 'oct-models-fr90-mega-1-mega-2'

    predict_3d, _res, _blur, _noise, _distortion_ids = load_3d_perf_prediction(result_id)
    pc3 = Partials(_res, _blur, _noise, func_3d_eval=predict_3d)

    pc3_x, pc3_y, pc3_z = pc3.gradient
    _res, _blur, _noise = pc3.coords
    _rr, _bb, _nn = pc3.coords_3d()

    grad_plot(_rr, _bb, _nn, pc3_x, pc3_y, pc3_z)

