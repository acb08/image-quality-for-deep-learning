"""
Calculates the Akaike information criterion (AIC) for a fit to evaluate the performance of a model
"""

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.binomial_simulation import run_binomial_accuracy_experiment


def akaike_info_criterion(acc, n_trials, acc_predicted, num_parameters, distribution='binomial'):

    if distribution != 'binomial':
        raise NotImplementedError

    __, __, log_likelihood = binomial_likelihood(acc=acc,
                                                 num_trials=n_trials,
                                                 acc_predicted=acc_predicted)

    return 2 * num_parameters - 2 * log_likelihood


def binomial(k, n, p):
    combinatoric_coeff = np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial((n - k)))
    return combinatoric_coeff * p ** k * (1 - p) ** (n - k)


# def approx_binomial_prob(acc, n_bar, p):
#     k_bar = int(acc * n_bar)  # estimated number of successes
#     return binomial(k_bar, n_bar, p)


def binomial_likelihood(acc, num_trials, acc_predicted, eps=1e-10):

    num_trials = int(num_trials)

    k = acc * num_trials
    k_int = np.asarray(np.round(k), dtype=np.int64)

    if hasattr(num_trials, '__len__'):  # if num_trials is a vector, then k (num successes) should be an integer
        assert max(np.abs(k - k_int)) < eps
    else:
        num_trials = np.round(num_trials)
        num_trials = num_trials * np.ones_like(acc_predicted)
        num_trials = np.asarray(num_trials, dtype=np.int64)

    likelihoods = []
    log_likelihoods = []

    for i, k_val in enumerate(k_int):
        acc_val = acc_predicted[i]
        num_trials_val = num_trials[i]
        llh = binomial(k_val, num_trials_val, acc_val)
        likelihoods.append(llh)
        log_likelihoods.append(np.log(llh))

    total_log_likelihood = np.sum(log_likelihoods)

    return likelihoods, log_likelihoods, total_log_likelihood


def product(array):
    p = 1
    for val in array:
        p *= val
    return p


if __name__ == '__main__':
    # calculate the aic!

    _n = 80
    _k = np.arange(_n + 1)
    _p = np.linspace(0.1, 0.9, num=(_n + 1))
    _p_error = _p + 0.1 * _p**2

    _acc_sim = run_binomial_accuracy_experiment(_p, _n)
    _acc_sim_error = run_binomial_accuracy_experiment(_p_error, _n)

    _sim_likelihood, _sim_log_likelihood, _total_log_likelihood = binomial_likelihood(_acc_sim, _n, _p)
    _sim_error_likelihood, _sim_error_log_likelihood, _total_error_log_likelihood = binomial_likelihood(_acc_sim_error, _n, _p)

    _total_likelihood = product(_sim_likelihood)
    print(np.log(_total_likelihood), _total_log_likelihood)
    _total_error_likelihood = product(_sim_error_likelihood)
    print(np.log(_total_error_likelihood), _total_error_log_likelihood)

    plt.figure()
    plt.plot(_p, _acc_sim)
    plt.plot(_p, _acc_sim_error)
    plt.show()

    plt.figure()
    plt.plot(_p)
    plt.plot(_p_error)
    plt.show()

    plt.figure()
    plt.plot(_sim_likelihood)
    plt.plot(_sim_error_likelihood)
    plt.show()

    _aic = akaike_info_criterion(_acc_sim, n_trials=_n, acc_predicted=_p, num_parameters=5)
    _aic_error = akaike_info_criterion(_acc_sim_error, n_trials=_n,  acc_predicted=_p, num_parameters=5)
    _aic_error_2 = akaike_info_criterion(_acc_sim_error, n_trials=_n,  acc_predicted=_p_error, num_parameters=5)
    _aic_error_3 = akaike_info_criterion(_acc_sim, n_trials=_n,  acc_predicted=_p_error, num_parameters=5)

    # pmf = []
    # for _k_val in _k:
    #     pmf.append(binomial(_k_val, _n, _p))
    # pmf = np.asarray(pmf)
    #
    # pmf_check = binom.pmf(_k, _n, _p)
    #
    # plt.figure()
    # plt.plot(_k, pmf)
    # plt.plot(_k, pmf_check)
    # plt.show()
    #
    # print(np.sum(pmf))
    # print(np.sum(pmf_check))
    #
