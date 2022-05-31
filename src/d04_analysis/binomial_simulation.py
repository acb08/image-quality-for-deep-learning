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


def ideal_correlation(p_fit, total_trials, iterations=8):
    """

    Simulates a perfect binomial distributed accuracy trial across an underlying accuracy (i.e. success probability)
    p_fit. Measures the correlation between two i.i.d. experiments sharing the same underlying accuracy p_fit as well as
    the correlation between p_fit and each trial.

    Intended to simulate the correlation between models with identical accuracy being tested on i.i.d. datasets, where
    p_fit is the underlying accuracy as a function of distortion.

    :param p_fit: estimate of intrinsic accuracy/p-success.
    :param total_trials: number of total trials (i.e. number of images in test dataset), assumed o be spread evenly
    across p_fit
    :param iterations: number of times to simulate the experiment
    :return:
        mean_correlation: average correlation between accuracy of identical trials run independently.
        mean_fit_correlation: average correlation between accuracy and the estimated/fit probability that was used as
        the underlying probability for the binomial experiment. Essentially it's the correlation that would be observed
        between a perfect fit of the underlying accuracy and a perfect set of binomial distributions centered on the
        fit probabilities
        correlations: the correlations between i.i.d. experiments in each iteration in range(iterations)
        fit_correlations: the correlations between the fits and the experiments based on the fit for both independent
        trials in range(iterations).
    """

    num_trials_per_experiment = total_trials / len(p_fit)
    correlations = []
    fit_correlations = []

    for j in range(iterations):

        trial_accuracy_0 = run_binomial_accuracy_experiment(p_fit, num_trials_per_experiment)
        trial_accuracy_1 = run_binomial_accuracy_experiment(p_fit, num_trials_per_experiment)
        correlation = measure_accuracy_correlations(trial_accuracy_0, trial_accuracy_1)
        correlations.append(correlation)

        fit_correlation_0 = measure_accuracy_correlations(p_fit, trial_accuracy_0)
        fit_correlation_1 = measure_accuracy_correlations(p_fit, trial_accuracy_0)
        fit_correlations.append((fit_correlation_0, fit_correlation_1))

    mean_correlation = np.mean(correlations)
    mean_fit_correlation = np.mean(fit_correlations)

    return mean_correlation, mean_fit_correlation, correlations, fit_correlations


if __name__ == '__main__':

    _num_experiments = 20 * 20 * 21  # roughly the number of discrete points in the distortion space
    _p_max = 0.45
    _p_min = 0.05
    _p_success = linear_p_success(_num_experiments, _p_max, _p_min)

    _total_trials = 35000 * 20

    _mean_correlation, _mean_fit_correlation, _correlations, _fit_correlations = ideal_correlation(_p_success,
                                                                                                   _total_trials)

    print(_mean_correlation)
    print(_mean_fit_correlation)
    print(_correlations)
    print(_fit_correlations)
