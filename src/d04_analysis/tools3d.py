# import numpy as np
#
# from src.d04_analysis.analysis_functions import build_3d_field
#
# from src.d04_analysis.plot import plot_isosurf
#
# if __name__ == '__main__':
#     _x = np.random.randint(0, 10, 10000)
#     _y = np.random.randint(0, 10, 10000)
#     _z = np.random.randint(0, 10, 10000)
#     # _f = np.sqrt(_x**2 + _y++2 + _z**2)
#     # _f = np.sqrt(_x**2 + _y + _z)
#     _f = _x ** 2 + _y ** 2 + _z ** 2
#     _x_vals, _y_vals, _z_vals, _f_means, _full_extract = build_3d_field(_x, _y, _z, _f)
#     plot_isosurf(_f_means, _x_vals, _y_vals, _z_vals, level=25)
