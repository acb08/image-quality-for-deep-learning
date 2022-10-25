import numpy as np
import matplotlib.pyplot as plt


class CompositeProps(object):  # TODO: figure out which methods to turn into dunders

    def __init__(self, properties,
                 parent_properties,
                 property_keys,
                 extractor,
                 iter_keys,
                 parent_iter_keys,
                 extractor_traverse_keys=None):
        self.props = properties
        self.parent_props = parent_properties
        self.property_keys = property_keys
        self.extractor = extractor
        self.iter_keys = iter_keys
        self.parent_iter_keys = parent_iter_keys
        self.traverse_keys = extractor_traverse_keys
        self.parameter_set = self.extractor(self.props, 'distortion_params', self.iter_keys)

    def effective(self, property_key):
        prop = self.extractor(self.props, property_key, self.iter_keys, self.traverse_keys)
        parent_prop = self.extractor(self.parent_props, property_key, self.parent_iter_keys, self.traverse_keys)
        diff = np.asarray(prop) - np.asarray(parent_prop)
        effective_property = parent_prop - diff
        return effective_property

    def all_effective_props(self):
        all_effective = {}
        for property_key in self.property_keys:
            all_effective[property_key] = self.effective(property_key)
        return all_effective

    def all_effective_keys(self):
        return self.property_keys

    def basic(self, property_key):
        prop = self.extractor(self.props, property_key, self.iter_keys, self.traverse_keys)
        return np.asarray(prop)

    def parent(self, property_key):
        parent_prop = self.extractor(self.parent_props, property_key, self.parent_iter_keys, self.traverse_keys)
        return np.asarray(parent_prop)

    def blur_values(self):
        stds = []
        for params in self.parameter_set:
            stds.append(params[2][1])
        return np.asarray(stds)

    def noise_values(self):
        _noise_means = []
        for params in self.parameter_set:
            _noise_means.append(params[3][1])
        return np.asarray(_noise_means)

    def res_values(self):
        _res_fractions = []
        for params in self.parameter_set:
            _res_fractions.append(params[1][1])
        return np.asarray(_res_fractions)


def conditional_mean_accuracy(labels, predicts, condition_array):

    """
    Returns mean accuracy as for each unique value in condition array.

    Note: switched the order of outputs to be x, y rather than y, x
    """

    conditions = np.unique(condition_array)
    conditioned_accuracies = np.zeros(len(conditions))

    counter = 0
    for condition in conditions:
        condition_labels = labels[np.where(condition_array == condition)]
        condition_predicts = predicts[np.where(condition_array == condition)]
        conditioned_accuracies[counter] = (
            np.mean(np.equal(condition_labels, condition_predicts)))
        counter += 1

    return conditions, conditioned_accuracies


def metric_perf_corrcoeff(performance, metric, hist):
    """
    Calculates the Pearson corelation coefficient between performance and metric
    using hist to appropriately weight the observations
    """

    cov = np.cov(performance, metric, fweights=hist)
    diag = np.atleast_2d(np.diag(cov))
    dSqr = diag.T * diag
    corr = cov / np.sqrt(dSqr)

    return corr[0, 1]


def top_k_accuracy_vector(ground_truth, predictions):
    """
    ground_truth: array of length n containing ground truth labels
    predictions: k x n array containing top-k predictions for n samples
    returns: top k accuracy, where k is specified by shape of the predictions array.

    """

    broadcaster = np.multiply(ground_truth, np.ones_like(predictions))
    comparison = np.equal(broadcaster, predictions)

    return comparison[0, :], np.sum(comparison, axis=0)




