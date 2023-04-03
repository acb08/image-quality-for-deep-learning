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


def binomial_likelihood(acc, num_trials, acc_predicted, eps=1e-10):

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


def sim_val_binomial():

    total_experiments = 4000
    n_trials = 100 + np.random.randint(0, 51, total_experiments)

    p_success_true = 0.35 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, num=total_experiments))
    error = np.linspace(0.1, 0.5, num=total_experiments) ** 3
    error_coefficients = np.linspace(0, 2, num=10)

    simulated_result = run_binomial_accuracy_experiment(p_success=p_success_true,
                                                        num_trials_per_experiment=n_trials)

    akaike_scores = []

    for coeff in error_coefficients:
        total_error = error * coeff
        p_success_predicted = p_success_true + total_error
        akaike_score = akaike_info_criterion(acc=simulated_result,
                                             n_trials=n_trials,
                                             acc_predicted=p_success_predicted,
                                             num_parameters=5)
        akaike_scores.append(akaike_score)

    fig, axes = plt.subplots()
    axes.plot(error_coefficients, akaike_scores)
    axes.set_xlabel('error coefficient')
    axes.set_ylabel('Akaike information criterion score')
    plt.show()


if __name__ == '__main__':

    sim_val_binomial()
