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


def accuracy_v_entropy(image_entropies, labels, predicts,
                       bin_specifier=None,
                       eps=0.001):
    """

    Function to calculate model accuracy as a fuction of image entropy

    inputs:
        image_entropies: 1D array of image entropies
        labels: image truth labels
        predicts: image labels predicted
        bin_specifier: integer specifying the number of entropy histogram bins to be
        generated or a string to specify the histogram edge algorithm to be used
        by numpy.historgram_bin_edges function
        eps: specifies amount above max(image_entropies) to be used in the range of
        numpy.histogram_bin_edges function. If eps = 0, the np.digitize function
        will assign the largest entropy value to the right_most histogram bin
        edge and there will be a mismatch between bin indices and histogram indices
        (i.e. setting eps > 0 is basically a bug fix)

    returns:
        acc_v_bin_index: average accuracy of the model at each for images
        in the corresponding entropy bin
        occupied_entropy_bins: entropy bin left edge values for occupied bins
        occupied_bin_left_edge_indices: entropy bin left edge indices for
        occupied bins
        entropy_bins: all entropy bins
        entropy_hist: histogram across all entropy bins

    """

    # get entropy bin edges using whichever method is specified. Default is
    # Freedman-Diaconis
    hist_range = (np.min(image_entropies), np.max(image_entropies) + eps)
    if bin_specifier == None:
        entropy_hist, entropy_bins = np.histogram(image_entropies,
                                                  bins='fd',
                                                  range=hist_range,
                                                  density=False)
    else:
        entropy_hist, entropy_bins = np.histogram(image_entropies,
                                                  bins=bin_specifier,
                                                  range=hist_range,
                                                  density=False)

    image_entropy_bin_indices = np.digitize(image_entropies, entropy_bins)

    # next get the conditional accuracy for images assigned to each
    # respective entropy bin
    acc_vs_bin_index, image_entropy_bin_indices_used = conditional_mean_accuracy(
        labels, predicts, image_entropy_bin_indices)

    occupied_entropy_bins = entropy_bins[image_entropy_bin_indices_used - 1]
    occupied_bin_left_edge_indices = image_entropy_bin_indices_used - 1

    return (acc_vs_bin_index, occupied_entropy_bins, occupied_bin_left_edge_indices,
            entropy_bins, entropy_hist)


def plots_n_correlations(image_entropies, labels, predicts,
                         folder=None,
                         model=None,
                         dataset=None,
                         data_labels=None,
                         entropy_type=None,
                         thresh=0.01):
    """
    Local function
    """

    acc_v_entropy, entropy_bins_used, entropy_bin_indices_used, bins, hist = (
        accuracy_v_entropy(image_entropies, labels, predicts))

    plt.figure()
    plt.plot(entropy_bins_used, acc_v_entropy)
    xlabel = 'Effective ' + str(entropy_type) + ' Entropy'
    plt.xlabel(xlabel)
    plt.ylabel('Average Accuracy')
    title = str(model) + ' Model, ' + str(dataset)
    plt.title(title)
    name = title + ' ' + str(entropy_type) + '.png'
    plt.savefig(name)

    histNormed = hist / np.sum(hist)
    acc_v_ent_thresh = acc_v_entropy[np.where(
        histNormed[entropy_bin_indices_used] >= thresh)]
    ent_bins_thresh = entropy_bins_used[np.where(
        histNormed[entropy_bin_indices_used] >= thresh)]
    plt.figure()
    plt.plot(ent_bins_thresh, acc_v_ent_thresh)
    xlabel = 'Effective ' + str(entropy_type) + ' Entropy'
    plt.xlabel(xlabel)
    plt.ylabel('Average Accuracy')
    title = str(model) + ' Model, ' + str(dataset) + \
            ' (bin weight >' + str(thresh) + ')'
    plt.title(title)
    name = (str(model) + ' Model, ' + str(dataset) + ' ' + str(entropy_type)
            + ' thresh (' + str(thresh) + ').png')
    # plt.savefig(name)

    rho = metric_perf_corrcoeff(acc_v_entropy, entropy_bins_used,
                                hist[entropy_bin_indices_used])
    print(model, 'model,', dataset, 'dataset,',
          entropy_type, 'entropy, rho =', rho)

    return rho


def top_k_accuracy_vector(ground_truth, predictions):
    """
    ground_truth: array of length n containing ground truth labels
    predictions: k x n array containing top-k predictions for n samples
    returns: top k accuracy, where k is specified by shape of the predictions array.

    """

    broadcaster = np.multiply(ground_truth, np.ones_like(predictions))
    comparison = np.equal(broadcaster, predictions)

    return comparison[0, :], np.sum(comparison, axis=0)




