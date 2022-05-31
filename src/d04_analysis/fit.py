import numpy as np


def fit_hyperplane(x, y, add_bias=True):

    """
    Fits an m-dimensional hyperplane w to x and y using singular value decomposition, where w is the least squares fit
    for the system wx=y, x is an m-by-n matrix, and y is an m-by-1 column vector. If add_bias is True, then x is
    augmented to be an n-by-(m+1) and w becomes an (m+1)-by-1 column vector.
    """

    if add_bias:
        n, m = np.shape(x)
        bias_col = np.ones((n, 1))
        x = np.append(x, bias_col, axis=1)

    ata = np.matmul(x.T, x)
    atb = np.matmul(x.T, y)

    u, s, v_t = np.linalg.svd(ata)
    s_inv = np.diag((1 / s))
    w = np.matmul(v_t.T, np.matmul(s_inv, np.matmul(u.T, atb)))

    return w


def eval_linear_fit(w, x, y, add_bias=True):
    # if add_bias:
    #     n, m = np.shape(x)
    #     bias_col = np.ones((n, 1))
    #     x = np.append(x, bias_col, axis=1)
    # y_predict = np.matmul(x, w)
    y_predict = linear_predict(w, x, add_bias=add_bias)
    correlation = np.corrcoef(np.ravel(y_predict), np.ravel(y))[0, 1]
    return correlation


def linear_predict(w, x, add_bias=True):
    if add_bias:
        n, m = np.shape(x)
        bias_col = np.ones((n, 1))
        x = np.append(x, bias_col, axis=1)
    y_predict = np.matmul(x, w)
    return y_predict


# if __name__ == '__main__':
#
#     n_pts = 20
#     slope = 2
#     bias = 2
#     y = []
#     x = 10 * np.random.rand(n_pts).reshape((n_pts, 1))
#     y = slope * x + bias + np.random.randn(n_pts).reshape((n_pts, 1))
#     fit = fit_hyperplane(x, y, add_bias=True)
#     x_plot = np.linspace(0, 10, 100)
#     y_fit = fit[0] * x_plot + fit[1]
#
#     plt.figure()
#     plt.scatter(x, y)
#     plt.plot(x_plot, y_fit)
#     plt.show()
#
#     X = np.random.randint(0, 10, (20, 3))
#     X = np.append(X, np.ones((20, 1)), axis=1)
#     w_true = np.random.rand(4).reshape((4, 1))
#     Y = np.matmul(X, w_true)
#     Y = Y + np.random.randn(*np.shape(Y))
#     w_fit = fit_hyperplane(X, Y, add_bias=False)





