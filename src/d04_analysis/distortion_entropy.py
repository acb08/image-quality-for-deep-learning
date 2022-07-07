import wandb
from src.d00_utils.definitions import STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME, \
    STANDARD_ENTROPY_PROPERTIES_FILENAME, STANDARD_DATASET_FILENAME, WANDB_PID, ROOT_DIR, REL_PATHS
from src.d04_analysis.distortion_performance import DistortedDataset
from src.d00_utils.functions import load_wandb_data_artifact, get_config, dict_val_lists_to_arrays
from src.d04_analysis.measure_entropy_properties import get_entropy_artifact_name, get_effective_entropy_equiv
from src.d04_analysis.analysis_functions import extract_embedded_vectors, conditional_mean_entropy, \
    conditional_extract_2d, build_3d_field
from src.d04_analysis.fit import fit_hyperplane, eval_linear_fit
from src.d04_analysis.plot import plot_1d_linear_fit, AXIS_LABELS, plot_2d, plot_2d_linear_fit
import numpy as np
import argparse
from pathlib import Path


class DistortionEntropyProperties(DistortedDataset):

    def __init__(self, run, dataset_id, convert_to_std=True, identifier=None):
        self.convert_to_std = convert_to_std
        self.identifier = identifier
        self.dataset, self.entropy, self.effective_entropy = load_dataset_and_entropy(run, dataset_id)
        DistortedDataset.__init__(self, self.dataset, convert_to_std=self.convert_to_std)
        self.entropy_function_tags = self.entropy['entropy_function_tags']
        self.entropy_props = extract_embedded_vectors(self.entropy,
                                                      intermediate_keys=['shard_entropy_properties'],
                                                      target_keys=self.entropy_function_tags,
                                                      return_full_dict=True)
        self.entropy_props = dict_val_lists_to_arrays(self.entropy_props)
        self.effective_entropy_ids = [get_effective_entropy_equiv(entropy_function_tag) for
                                      entropy_function_tag in self.entropy_function_tags]
        self.effective_entropy_props = extract_embedded_vectors(self.effective_entropy,
                                                                intermediate_keys=[
                                                                    'shard_effective_entropy_properties'],
                                                                target_keys=self.effective_entropy_ids,
                                                                return_full_dict=True)
        self.effective_entropy_props = dict_val_lists_to_arrays(self.effective_entropy_props)

    def __repr__(self):
        return str(self.identifier)

    def conditional_effective_entropy(self, effective_entropy_id, distortion_id):
        return conditional_mean_entropy(self.effective_entropy_props[effective_entropy_id],
                                        self.distortions[distortion_id])


def load_dataset_and_entropy(run, dataset_id,
                             dataset_alias='latest',
                             entropy_alas='latest',
                             effective_entropy_alias='latest',
                             dataset_filename=STANDARD_DATASET_FILENAME):

    entropy_artifact_name = get_entropy_artifact_name(dataset_id, effective=False)
    effective_entropy_artifact_name = get_entropy_artifact_name(dataset_id, effective=True)

    if ':' not in dataset_id:
        dataset_id = f'{dataset_id}:{dataset_alias}'
    entropy_artifact_id = f'{entropy_artifact_name}:{entropy_alas}'
    effective_entropy_artifact_id = f'{effective_entropy_artifact_name}:{effective_entropy_alias}'

    dataset_dir, dataset = load_wandb_data_artifact(run, dataset_id, dataset_filename)
    entropy_dir, entropy_properties = load_wandb_data_artifact(run, entropy_artifact_id,
                                                               STANDARD_ENTROPY_PROPERTIES_FILENAME)
    effective_entropy_dir, effective_entropy_properties = load_wandb_data_artifact(
        run, effective_entropy_artifact_id, STANDARD_EFFECTIVE_ENTROPY_PROPERTIES_FILENAME)

    return dataset, entropy_properties, effective_entropy_properties


def get_distortion_entropy_properties(dataset_id=None, identifier=None, convert_to_std=True, config=None):

    if not dataset_id and not identifier:
        dataset_id = config['dataset_id']
        identifier = config['identifier']
        convert_to_std = config['convert_to_std']

    with wandb.init(project=WANDB_PID, job_type='analyze_distortion_entropy_properties') as run:
        output_dir = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['entropy'], dataset_id)
        if not output_dir.is_dir():
            Path.mkdir(output_dir, parents=True)

        distortion_entropy_properties = DistortionEntropyProperties(run, dataset_id, convert_to_std=convert_to_std,
                                                                    identifier=identifier)

    return distortion_entropy_properties, output_dir


def analyze_entropy_1d(dist_entropy_props,
                       directory=None,
                       log_file=None):

    for effective_entropy_id in dist_entropy_props.effective_entropy_ids:
        for distortion_id in dist_entropy_props.distortion_ids:
            distortion_vals, mean_entropy_effective_entropy_vals, fit, fit_correlations, __ = (
                get_distortion_entropy_relationship_1d(dist_entropy_props, effective_entropy_id, distortion_id,
                                                       log_file=log_file, add_bias=True))
            plot_1d_linear_fit(distortion_vals, mean_entropy_effective_entropy_vals, fit, distortion_id,
                               result_identifier=effective_entropy_id, ylabel=AXIS_LABELS['effective_entropy'],
                               directory=directory)


def get_distortion_entropy_relationship_1d(dist_entropy_props, effective_entropy_id, distortion_id, log_file=None,
                                           add_bias=True):

    raw_correlation = np.corrcoef(dist_entropy_props.effective_entropy_props[effective_entropy_id],
                                  dist_entropy_props.distortions[distortion_id])[0][1]

    distortion_vals, mean_effective_entropies = dist_entropy_props.conditional_effective_entropy(
        effective_entropy_id, distortion_id)

    fit_coefficients = fit_hyperplane(np.atleast_2d(distortion_vals).T,
                                      np.atleast_2d(mean_effective_entropies).T,
                                      add_bias=add_bias)
    mean_correlation = eval_linear_fit(fit_coefficients,
                                       np.atleast_2d(distortion_vals).T,
                                       np.atleast_2d(mean_effective_entropies).T)

    print(f'{effective_entropy_id} {distortion_id} raw correlation: {raw_correlation}', file=log_file)
    print(f'{effective_entropy_id} {distortion_id} linear fit: ', fit_coefficients, file=log_file)
    print(f'{effective_entropy_id} {distortion_id} linear fit correlation: ', mean_correlation, file=log_file)

    return distortion_vals, mean_effective_entropies, fit_coefficients, mean_correlation, raw_correlation


def analyze_entropy_2d(dist_entropy_props,
                       distortion_combinations=((0, 1), (1, 2), (0, 2)),
                       directory=None,
                       log_file=None,
                       perform_fit=False):

    distortion_ids = dist_entropy_props.distortion_ids

    for effective_entropy_id in dist_entropy_props.effective_entropy_ids:
        for i, (idx_0, idx_1) in enumerate(distortion_combinations):
            x_id, y_id = distortion_ids[idx_0], distortion_ids[idx_1]

            x_vals, y_vals, entropy_means, fit, corr, distortion_arr = get_distortion_entropy_relationship_2d(
                dist_entropy_props, effective_entropy_id, x_id, y_id, log_file=log_file, add_bias=True,
                perform_fit=perform_fit)

            plot_2d(x_vals, y_vals, entropy_means, x_id, y_id,
                    result_identifier=effective_entropy_id,
                    axis_labels='effective_entropy_default',
                    directory=directory)
            if perform_fit:
                plot_2d_linear_fit(distortion_arr, entropy_means, fit, x_id, y_id,
                                   result_identifier=effective_entropy_id, axis_labels='effective_entropy_default',
                                   directory=directory)


def get_distortion_entropy_relationship_2d(dist_entropy_props, effective_entropy_id, x_id, y_id, log_file=None,
                                           add_bias=True, perform_fit=False):

    entropy = dist_entropy_props.effective_entropy_props[effective_entropy_id]
    x = dist_entropy_props.distortions[x_id]
    y = dist_entropy_props.distortions[y_id]

    x_values, y_values, entropy_means, vector_data_extract = conditional_extract_2d(x, y, entropy)

    fit_coefficients = None
    correlation = None
    distortion_param_array = None

    if perform_fit:
        distortion_param_array = vector_data_extract['param_array']
        entropy_array = vector_data_extract['performance_array']
        fit_coefficients = fit_hyperplane(distortion_param_array, entropy_array, add_bias=add_bias)
        correlation = eval_linear_fit(fit_coefficients, distortion_param_array, entropy_array, add_bias=add_bias)

        print(f'{effective_entropy_id} {x_id} {y_id} linear fit: ', fit_coefficients, file=log_file)
        print(f'{effective_entropy_id} {x_id} {y_id} linear fit correlation: ', correlation, '\n', file=log_file)

    return x_values, y_values, entropy_means, fit_coefficients, correlation, distortion_param_array


def analyze_entropy_3d(dist_entropy_props, log_file=None):

    entropy_fields_3d = {}
    x_id, y_id, z_id = dist_entropy_props.distortion_ids

    for effective_entropy_id in dist_entropy_props.effective_entropy_ids:
        x_vals, y_vals, z_vals, entropy_3d = get_distortion_entropy_relationship_3d(dist_entropy_props,
                                                                                    effective_entropy_id,
                                                                                    x_id=x_id, y_id=y_id, z_id=z_id,
                                                                                    log_file=log_file, add_bias=True)
        entropy_fields_3d[effective_entropy_id] = {
            x_id: x_vals,
            y_id: y_vals,
            z_id: z_vals,
            'entropy_3d': entropy_3d
        }

    return entropy_fields_3d


def get_distortion_entropy_relationship_3d(dist_entropy_props, effective_entropy_id, x_id='res', y_id='blur',
                                           z_id='noise', log_file=None, add_bias=True, perform_fit=False):

    entropy = dist_entropy_props.effective_entropy_props[effective_entropy_id]
    x = dist_entropy_props.distortions[x_id]
    y = dist_entropy_props.distortions[y_id]
    z = dist_entropy_props.distortions[z_id]

    x_vals, y_vals, z_vals, entropy_3d, distortion_array, entropy_array, __ = build_3d_field(x, y, z, entropy,
                                                                                             data_dump=True)

    if perform_fit:
        fit_coefficients = fit_hyperplane(distortion_array, entropy_array, add_bias=add_bias)
        correlation = eval_linear_fit(fit_coefficients, distortion_array, entropy_array, add_bias=add_bias)

        print(f'{effective_entropy_id} {x_id} {y_id} {z_id} linear fit: ', fit_coefficients, file=log_file)
        print(f'{effective_entropy_id} {x_id} {y_id} {z_id} linear fit correlation: ', correlation, '\n', file=log_file)

    return x_vals, y_vals, z_vals, entropy_3d


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='distortion_entropy_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'distortion_entropy_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)
    perform_2d_fit = run_config['perform_2d_fit']
    perform_3d_fit = run_config['perform_3d_fit']

    props, result_dir = get_distortion_entropy_properties(config=run_config)

    with open(Path(result_dir, 'result_log.txt'), 'w') as output_file:
        analyze_entropy_1d(props, directory=result_dir, log_file=output_file)
        analyze_entropy_2d(props, perform_fit=perform_2d_fit, directory=result_dir, log_file=output_file)
        analyze_entropy_3d(props, log_file=output_file)

