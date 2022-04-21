import numpy as np


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


def get_class_accuracies(labels, predicts):
    """
    Extracts mean accuracy of predicts for each unique label in labels
    """

    classes = np.unique(labels)
    class_accuracies = []
    for _class in classes:
        class_predicts = predicts[np.where(labels == _class)]
        class_accuracy = np.mean(np.equal(class_predicts, _class))
        class_accuracies.append(class_accuracy)

    return classes, np.asarray(class_accuracies)


def conditional_mean_accuracy(labels, predicts, condition_array, per_class=False):

    """
    Returns mean accuracy for each unique value in condition array. If per_class == True, returns mean per_class
    accuracy for each unique value in condition array

    Note: switched the order of outputs to be x, y rather than y, x
    """

    conditions = np.unique(condition_array)
    conditioned_accuracies = np.zeros(len(conditions))

    for i, condition in enumerate(conditions):
        condition_labels = labels[np.where(condition_array == condition)]
        condition_predicts = predicts[np.where(condition_array == condition)]
        if per_class:
            __, class_accuracies = get_class_accuracies(condition_labels, condition_predicts)
            conditioned_accuracies[i] = np.mean(class_accuracies)
        if not per_class:
            conditioned_accuracies[i] = (
                np.mean(np.equal(condition_labels, condition_predicts)))

    return conditions, conditioned_accuracies


def conditional_mean_entropy(entropy, condition_array):

    conditions = np.unique(condition_array)
    conditional_entropies = np.zeros(len(conditions))

    for i, condition in enumerate(conditions):
        conditional_entropies[i] = np.mean(entropy[np.where(condition_array == condition)])

    return conditions, conditional_entropies


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


def conditional_extract_2d(x, y, z):
    """
    x: array of length N containing j unique values
    y: array of length N containing k unique values
    z: array of length N, with values that will be sorted into a 2D j-by-k
    array, with indices (alpha, beta), where 0 <= alpha <= j-1, 0 <= beta <= k-1,
    where each (alpha, beta) represents a unique combination
    of the unique values of x and y. In other words, lets imagine that z is a
    2d function of two variables sigma and lambda, and that we have N samples
    of z, with each sample corresponding to a pair values sigma and lambda.
    This function extracts the relevant values of z for each unique
    (sigma, lambda) pair.

    returns:

        x_values: numpy array, where x_values[alpha] represents the alpha-th unique
        value of x
        y_values: numpy array, where y_values[beta] represents the beta-th unique
        value of y
        z_means: j x k array, where z_means[alpha, beta] is the mean of z
        where x == x_values[alpha] and y == y_values[beta]
        extracts: dictionary, where keys are tuples (alpha, beta) and values are
        1D numpy arrays of z values where x == alpha and y == beta

    """

    x_values = np.unique(x)
    y_values = np.unique(y)
    z_means = np.zeros((len(x_values), len(y_values)))

    param_array = []  # for use in curve fits
    performance_array = []  # for use in svd

    for x_counter, x_val in enumerate(x_values):
        x_inds = np.where(x == x_val)
        for y_counter, y_val in enumerate(y_values):
            y_inds = np.where(y == y_val)
            z_inds = np.intersect1d(x_inds, y_inds)
            z_means[x_counter, y_counter] = np.mean(z[z_inds])
            param_array.append([x_val, y_val])
            performance_array.append(z_means[x_counter, y_counter])

    # full extract arrays written out this way for use in svd.
    vector_data_extract = {
        'param_array': np.asarray(param_array),
        'performance_array': np.atleast_2d(performance_array).T
    }

    return x_values, y_values, z_means, vector_data_extract





