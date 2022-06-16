import numpy as np

from src.d04_analysis.fit import fit_hyperplane, eval_linear_fit


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


def create_identifier(perf_results, dim_tag=None, dataset_identifier=None):
    """
    Creates identifier from a list of model performance results. Intended for use in plotting different
    sets of results together (i.e. built to be called by plot_results_together())
    """
    identifier = None

    for perf_result in perf_results:
        if not identifier:
            identifier = str(perf_result)
        else:
            identifier = f'{identifier}_{str(perf_result)}'

    if dim_tag:
        identifier = f'{dim_tag}_{identifier}'

    if dataset_identifier:
        identifier = f'{dataset_identifier}_{identifier}'

    return identifier


def build_3d_field(x, y, z, f, data_dump=False):
    """
    x: array of length N containing nx unique values
    y: array of length N containing ny unique values
    z: array of length N containing nz unique values
    f: array of length N, with values that will be sorted into a 3D (nx, ny, nz)-shape
    array, with indices (i, j, k), where 0 <= i <= nx-1, 0 <= j <= ny-1, 0 <= k <= nz-1,
    where each (i, j, k) represents a unique combination
    of the unique values of x, y, and z. In other words, lets imagine that f is a
    3d function of two variables sigma and lambda, and that we have N samples
    of z, with each sample corresponding to a pair values sigma and lambda.
    This function extracts the relevant values of z for each unique
    (sigma, lambda) pair.

    returns:

        x_values: numpy array, where x_values[alpha] represents the alpha-th unique
        value of x
        y_values: numpy array, where y_values[beta] represents the beta-th unique
        value of y
        z_values: numpy array, where z_values[gamma] represents the gamma-th unique
        value of z
        f_means: nx x ny x nz array, where f_means[alpha, beta, gamma] is the mean of f
        where x == x_values[alpha], y == y_values[beta], and z == y_values[gamma]
        parameter_array: (nx * ny * nz) x 3 array, where each row contains a unique (x_value, y_value, z_value)
        combination. Intended for use in fitting performance prediction functions using SVD.
        performance_array: (nx * ny * nz) x 1 array, where each element represents the mean the mean performance where
        x == x_values[alpha], y == y_values[beta], and z == y_values[gamma]. Contains the same information as f_means
        arranged differently. Intended for use in fitting performance prediction functions using SVD.
        extracts: dictionary, where keys are tuples (alpha, beta, gamma) and values are
        1D numpy arrays of f values where x == x_values[alpha], y == y_values[beta], z == z_values[gamma]

    """

    full_extract = {}  # diagnostic
    x_values = np.unique(x)
    y_values = np.unique(y)
    z_values = np.unique(z)
    f_means = np.zeros((len(x_values), len(y_values), len(z_values)))  # note: to re-create a similar array with
    # np.meshgrid, we need to specify indexing='ij' indexing rather than the default 'xy' cartesian indexing

    parameter_array = []  # for use in curve fits
    performance_array = []  # for use in svd

    for i, x_val in enumerate(x_values):
        x_inds = np.where(x == x_val)
        for j, y_val in enumerate(y_values):
            y_inds = np.where(y == y_val)
            for k, z_val in enumerate(z_values):
                z_inds = np.where(z == z_val)
                xy_inds = np.intersect1d(x_inds, y_inds)
                xyz_inds = np.intersect1d(xy_inds, z_inds)

                full_extract[(x_val, y_val, z_val)] = f[xyz_inds]
                f_means[i, j, k] = np.mean(f[xyz_inds])
                parameter_array.append([x_val, y_val, z_val])
                performance_array.append(f_means[i, j, k])

    if data_dump:
        parameter_array = np.asarray(parameter_array, dtype=np.float32)
        performance_array = np.atleast_2d(np.asarray(performance_array, dtype=np.float32)).T
        return x_values, y_values, z_values, f_means, parameter_array, performance_array, full_extract
    else:
        return f_means


def get_distortion_perf_2d(model_performance, x_id, y_id, add_bias=True, log_file=None):
    result_name = str(model_performance)

    accuracy_vector = model_performance.top_1_vec
    x = model_performance.distortions[x_id]
    y = model_performance.distortions[y_id]
    x_values, y_values, accuracy_means, vector_data_extract = conditional_extract_2d(x, y, accuracy_vector)

    distortion_param_array = vector_data_extract['param_array']
    performance_array = vector_data_extract['performance_array']
    fit_coefficients = fit_hyperplane(distortion_param_array, performance_array, add_bias=add_bias)
    correlation = eval_linear_fit(fit_coefficients, distortion_param_array, performance_array, add_bias=add_bias)

    print(f'{result_name} {x_id} {y_id} linear fit: ', fit_coefficients, file=log_file)
    print(f'{result_name} {x_id} {y_id} linear fit correlation: ', correlation, '\n', file=log_file)

    return x_values, y_values, accuracy_means, fit_coefficients, correlation, distortion_param_array


def get_distortion_perf_1d(model_performance, distortion_id, log_file=None, add_bias=True, per_class=False):
    result_name = str(model_performance)
    distortion_vals, mean_accuracies = model_performance.conditional_accuracy(distortion_id, per_class=per_class)

    fit_coefficients = fit_hyperplane(np.atleast_2d(distortion_vals).T,
                                      np.atleast_2d(mean_accuracies).T,
                                      add_bias=add_bias)

    correlation = eval_linear_fit(fit_coefficients,
                                  np.atleast_2d(distortion_vals).T,
                                  np.atleast_2d(mean_accuracies).T)

    print(f'{result_name} {distortion_id} (per_class = {per_class}) linear fit: ', fit_coefficients, file=log_file)
    print(f'{result_name} {distortion_id} (per_class = {per_class}) linear fit correlation: ', correlation, '\n',
          file=log_file)

    return distortion_vals, mean_accuracies, fit_coefficients, correlation


def measure_log_perf_correlation(performance_result_pair, distortion_ids, log_file=None):

    if len(performance_result_pair) != 2:
        raise Exception('performance_result_pair must be length 2')

    if type(performance_result_pair) == dict:
        performance_result_pair = list(performance_result_pair.values())

    correlation = np.corrcoef(np.ravel(performance_result_pair[0]),
                              np.ravel(performance_result_pair[1]))[0, 1]
    print(f'{distortion_ids} pairwise result correlation: {correlation} \n', file=log_file)

    return correlation


def flatten(f, x_vals, y_vals, z_vals, flatten_axis=0):
    """
    :param f: 3d array of shape (len(x_vals), len(y_vals), len(z_vals)), presumed to be a function f(x, y, z).
    Note this shape corresponds to the array shape returned by np.meshgrid(x_vals, y_vals, z_vals, indexing='ij').
    :param x_vals: independent variable values on to axis-0
    :param y_vals: independent variable values on to axis-1
    :param z_vals: independent variable values on to axis-2
    :param flatten_axis: the axis along which f will be averaged (i.e. the axis to be removed by averaging out). Must be
    0, 1, or 2.
    :return: 2d array corresponding of f averaged along flatten axis, 1d array containing first remaining axis not
    eliminated by averaging (i.e. x_vals or y_vals), and 1d array containing the second axis not eliminated by
    averaging (i.e. y_vals or z_vals)
    """

    if flatten_axis not in range(3):
        raise ValueError

    f_2d = np.mean(f, axis=flatten_axis)
    remaining_axes = keep_2_of_3(a=x_vals, b=y_vals, c=z_vals, discard_idx=flatten_axis)

    return f_2d, remaining_axes[0], remaining_axes[1]


def keep_2_of_3(a=None, b=None, c=None, discard_idx=0):
    """
    Intended for use with flatten() function to keep track of the remaining axes/labels when a 3d array is flattened to
    its 2d average along one of its 3 axes. Essentially, if we average along the axes 2 (i.e z), keep_2_of_3() takes
    the original 3 axes (or their labels) and returns the two remaining, in this case x and y.

    NOTE: assumes that the 3d array in array in question uses 'ij' rather than 'xy' indexing.
    """

    if b is None and c is None:
        if len(a) != 3:
            raise Exception('if b and c are None, a must be length 3')
        starters = a
    # elif hasattr(a, '__len__'):
    #     raise Exception('a must not have attribute __len__ if b and c are not None')
    else:
        starters = (a, b, c)
        # if None in starters:
        #     raise Exception('keep_2_of_3 requires either len(a) == 3 with b and c None, or else a, b, and c must not '
        #                     'be None')

    keepers = []
    if discard_idx not in range(3):
        raise ValueError

    for i in range(3):
        if i != discard_idx:
            keepers.append(starters[i])

    return keepers


def sort_parallel(prediction, result):

    prediction = np.ravel(prediction)
    result = np.ravel(result)
    order = prediction.argsort()

    prediction = prediction[order]
    result = result[order]

    return prediction, result

