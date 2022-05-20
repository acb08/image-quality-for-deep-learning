import numpy as np
from numpy.random import default_rng

rng = default_rng()


def linear_p_success(num_experiments, p_max, p_min):

    x = np.arange(num_experiments)
    slope = (p_max - p_min) / num_experiments
    p = p_min + slope * x

    return p


def run_binomial_accuracy_experiment(p_success, num_trials_per_experiment):

    successes = rng.binomial(num_trials_per_experiment, p_success)
    mean_success = successes / num_trials_per_experiment

    return mean_success


def measure_accuracy_correlations(f0, f1):

    correlation = np.corrcoef(np.ravel(f0), np.ravel(f1))[0, 1]

    return correlation


if __name__ == '__main__':

    _num_experiments = 20 * 20 * 21  # roughly the number of discrete points in the distortion space
    _p_max = 0.5
    _p_min = 0.45
    _p_success = linear_p_success(80, _p_max, _p_min)
    _num_trials_per_experiment = 80

    _iterations = 10
    _correlations = []
    _center_prob_correlations = []

    for i in range(_iterations):

        _mean_accuracy_0 = run_binomial_accuracy_experiment(_p_success, _num_trials_per_experiment)
        _mean_accuracy_1 = run_binomial_accuracy_experiment(_p_success, _num_trials_per_experiment)
        _correlation = measure_accuracy_correlations(_mean_accuracy_0, _mean_accuracy_1)
        _cp_correlation_0 = measure_accuracy_correlations(_p_success, _mean_accuracy_0)
        _cp_correlation_1 = measure_accuracy_correlations(_p_success, _mean_accuracy_1)
        _center_prob_correlations.append((_cp_correlation_0, _cp_correlation_1))
        _correlations.append(_correlation)

    print(_correlations)
    print(_center_prob_correlations)
