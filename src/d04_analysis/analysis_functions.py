import numpy as np
import matplotlib.pyplot as plt


def extract_combine_shard_vector_data(data, target_keys, check_keys=False):
    """
    Extracts target vectors from a top-level dictionary keyed by shard ids (generally filenames), where each entry of
    for form data[shard_id] is another dict keyed with members of target_keys.

    Concatenates target vectors associated with each target_key in target_keys according to the order of the shard ids
    in data.

    Returns a dict containing concatenated target vectors keyed with their respective target keys and as well as a key
    check vectors for each target key to enable downstream checking of the shard id associated with each entry in the
    output target vectors.

    :param data: dict of the following form:
        {shard_id_0: {
            target_key_0: [],
            target_key_1: [],
            ...}
        shard_id_1: {
            target_key_0: []
            target_key_1: []
            ...}
        ...}

    :param target_keys: tuple or list
    :param check_keys: Bool

    :return: if check_keys, dict of the following form:
        {target_key_0: []
        target_key_0_key_check: []
        etc.}
    if not check_keys, dict of the following form:
        {target_key_0: []
        etc.}
    """

    shard_extracts = {}

    for target_key in target_keys:

        extract = []
        vector_file_key_checker = []

        for vector_file_key in data:
            shard_data = data[vector_file_key][target_key]
            extract.extend(shard_data)
            vector_keys = [vector_file_key] * len(shard_data)
            vector_file_key_checker.extend(vector_keys)

        shard_extracts[target_key] = extract
        if check_keys:
            shard_extracts[f'{target_key}_key_check'] = vector_file_key_checker

    return shard_extracts


def extract_embedded_vectors(data_dict,
                             intermediate_keys,
                             target_keys,
                             return_full_dict=False):

    """
    Extracts and builds lists from shard level data embedded in data_dict, where the target data is stored in a nested dict of the
    form target_dict = {
        shard_id_0: {
            target_key_0: [],
            target_key_1: [],
            etc.
        }
        shard_id_1: {
            target_key_0: []
            target_key_2: []
            etc.
        }
        etc.
    }.  target_dict is found at data_dict[intermediate_keys[0]][intermediate_keys[1]][etc.] The actual shard level data
    extraction is accomplished by extract_combine_shard_vector_data(*args)

    :param data_dict:
    :param intermediate_keys:
    :param target_keys:
    :param return_full_dict:
    :return: if return_full_dict, returns dict of the following form:
        {target_key_0: []
        target_key_0_key_check: []
        etc.}
    if not return_full_dict, returns list of the form
        [full_dict[target_key_0], full_dict[target_key_1], etc.], where each element full_dict[target_key_n] is itself
        a list.
    """

    target_dict = data_dict
    if intermediate_keys is not None:
        for intermediate_key in intermediate_keys:
            target_dict = target_dict[intermediate_key]

    extracted_data = extract_combine_shard_vector_data(target_dict, target_keys)

    if return_full_dict:
        return extracted_data

    else:
        distortion_vectors = []
        for target_key in target_keys:
            distortion_vectors.append(extracted_data[target_key])

        return distortion_vectors


# class CompositeProps(object):
#
#     def __init__(self, properties,
#                  parent_properties,
#                  property_keys,
#                  extractor,
#                  iter_keys,
#                  parent_iter_keys,
#                  extractor_traverse_keys=None):
#         self.props = properties
#         self.parent_props = parent_properties
#         self.property_keys = property_keys
#         self.extractor = extractor
#         self.iter_keys = iter_keys
#         self.parent_iter_keys = parent_iter_keys
#         self.traverse_keys = extractor_traverse_keys
#         self.parameter_set = self.extractor(self.props, 'distortion_params', self.iter_keys)
#
#     def effective(self, property_key):
#         prop = self.extractor(self.props, property_key, self.iter_keys, self.traverse_keys)
#         parent_prop = self.extractor(self.parent_props, property_key, self.parent_iter_keys, self.traverse_keys)
#         diff = np.asarray(prop) - np.asarray(parent_prop)
#         effective_property = parent_prop - diff
#         return effective_property
#
#     def all_effective_props(self):
#         all_effective = {}
#         for property_key in self.property_keys:
#             all_effective[property_key] = self.effective(property_key)
#         return all_effective
#
#     def all_effective_keys(self):
#         return self.property_keys
#
#     def basic(self, property_key):
#         prop = self.extractor(self.props, property_key, self.iter_keys, self.traverse_keys)
#         return np.asarray(prop)
#
#     def parent(self, property_key):
#         parent_prop = self.extractor(self.parent_props, property_key, self.parent_iter_keys, self.traverse_keys)
#         return np.asarray(parent_prop)
#
#     def blur_values(self):
#         stds = []
#         for params in self.parameter_set:
#             stds.append(params[2][1])
#         return np.asarray(stds)
#
#     def noise_values(self):
#         _noise_means = []
#         for params in self.parameter_set:
#             _noise_means.append(params[3][1])
#         return np.asarray(_noise_means)
#
#     def res_values(self):
#         _res_fractions = []
#         for params in self.parameter_set:
#             _res_fractions.append(params[1][1])
#         return np.asarray(_res_fractions)


def conditional_mean_accuracy(labels, predicts, condition_array):

    """
    Returns mean accuracy as for each unique value in condition array.

    Note: switched the order of outputs to be x, y rather than y, x
    """

    conditions = np.unique(condition_array)
    conditioned_accuracies = np.zeros(len(conditions))

    for i, condition in enumerate(conditions):
        condition_labels = labels[np.where(condition_array == condition)]
        condition_predicts = predicts[np.where(condition_array == condition)]
        conditioned_accuracies[i] = (
            np.mean(np.equal(condition_labels, condition_predicts)))

    return conditions, conditioned_accuracies


def conditional_mean_entropy(entropy, condition_array):

    conditions = np.unique(condition_array)
    conditional_entropies = np.zeros(len(conditions))

    for i, condition in enumerate(conditions):
        conditional_entropies[i] = np.mean(entropy[np.where(condition_array == condition)])

    return conditions, conditional_entropies


# def metric_perf_corrcoeff(performance, metric, hist):
#     """
#     Calculates the Pearson corelation coefficient between performance and metric
#     using hist to appropriately weight the observations
#     """
#
#     cov = np.cov(performance, metric, fweights=hist)
#     diag = np.atleast_2d(np.diag(cov))
#     dSqr = diag.T * diag
#     corr = cov / np.sqrt(dSqr)
#
#     return corr[0, 1]


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
    if bin_specifier is None:
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


# def plots_n_correlations(image_entropies, labels, predicts,
#                          folder=None,
#                          model=None,
#                          dataset=None,
#                          data_labels=None,
#                          entropy_type=None,
#                          thresh=0.01):
#     """
#     Local function
#     """
#
#     acc_v_entropy, entropy_bins_used, entropy_bin_indices_used, bins, hist = (
#         accuracy_v_entropy(image_entropies, labels, predicts))
#
#     plt.figure()
#     plt.plot(entropy_bins_used, acc_v_entropy)
#     xlabel = 'Effective ' + str(entropy_type) + ' Entropy'
#     plt.xlabel(xlabel)
#     plt.ylabel('Average Accuracy')
#     title = str(model) + ' Model, ' + str(dataset)
#     plt.title(title)
#     name = title + ' ' + str(entropy_type) + '.png'
#     plt.savefig(name)
#
#     histNormed = hist / np.sum(hist)
#     acc_v_ent_thresh = acc_v_entropy[np.where(
#         histNormed[entropy_bin_indices_used] >= thresh)]
#     ent_bins_thresh = entropy_bins_used[np.where(
#         histNormed[entropy_bin_indices_used] >= thresh)]
#     plt.figure()
#     plt.plot(ent_bins_thresh, acc_v_ent_thresh)
#     xlabel = 'Effective ' + str(entropy_type) + ' Entropy'
#     plt.xlabel(xlabel)
#     plt.ylabel('Average Accuracy')
#     title = str(model) + ' Model, ' + str(dataset) + \
#             ' (bin weight >' + str(thresh) + ')'
#     plt.title(title)
#     name = (str(model) + ' Model, ' + str(dataset) + ' ' + str(entropy_type)
#             + ' thresh (' + str(thresh) + ').png')
#     # plt.savefig(name)
#
#     rho = metric_perf_corrcoeff(acc_v_entropy, entropy_bins_used,
#                                 hist[entropy_bin_indices_used])
#     print(model, 'model,', dataset, 'dataset,',
#           entropy_type, 'entropy, rho =', rho)
#
#     return rho
#
#
# def top_k_accuracy_vector(ground_truth, predictions):
#     """
#     ground_truth: array of length n containing ground truth labels
#     predictions: k x n array containing top-k predictions for n samples
#     returns: top k accuracy, where k is specified by shape of the predictions array.
#
#     """
#
#     broadcaster = np.multiply(ground_truth, np.ones_like(predictions))
#     comparison = np.equal(broadcaster, predictions)
#
#     return comparison[0, :], np.sum(comparison, axis=0)




