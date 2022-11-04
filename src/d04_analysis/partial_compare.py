import numpy as np
import matplotlib.pyplot as plt


class Partials(object):

    def __init__(self, x, y, z, partial_func_x, partial_func_y, partial_func_z,
                 xlabel='x', ylabel='y', zlabel='z'):

        # coordinates
        self.x = x
        self.y = y
        self.z = z
        self._xx, self._yy, self._zz = np.meshgrid(x, y, z)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.axis_labels = (self.xlabel, self.ylabel, self.zlabel)

        # partial derivative functions (functions of (x, y, z) in general)
        self.partial_func_x = partial_func_x
        self.partial_func_y = partial_func_y
        self.partial_func_z = partial_func_z

        self.f_x, self.f_y, self.f_z = self._evaluate_partials()
        self.gradient_magnitude = self._grad_mag()

        self.gradient = (self.f_x, self.f_y, self.f_z)
        self.coords = (self.x, self.y, self.z)

        self.shape = np.shape(self.gradient_magnitude)

    def _evaluate_partials(self):

        f_x_eval = self.partial_func_x(self._xx, self._yy, self._zz)
        f_y_eval = self.partial_func_y(self._xx, self._yy, self._zz)
        f_z_eval = self.partial_func_y(self._xx, self._yy, self._zz)

        return f_x_eval, f_y_eval, f_z_eval

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
    return 0.5 * yy**2


def p_z(xx, yy, zz):
    return xx - yy + 0.2 * np.sqrt(zz)


if __name__ == '__main__':

    _x, _y, _z = np.linspace(0, 1, num=4), np.linspace(0, 1, num=4), np.linspace(0, 1, num=4)
    partial_compare = Partials(_x, _y, _z,
                               p_x, p_y, p_z)
    _xx, _yy, _zz = np.meshgrid(_x, _y, _z)

    _u, _v, _w = partial_compare.gradient
    grad_plot(_xx, _yy, _zz, _u, _v, _w)


