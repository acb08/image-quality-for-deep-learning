from src.d03_test.test_model import test_model
from src.d00_utils.functions import get_config
import argparse
from pathlib import Path
import copy


def extract_model_test_configs(multi_model_test_config):

    """
    Takes a test config containing a list of model artifact ids (model_artifact_ids) in place of a single model artifact
    id (model_artifact_id), extracts and returns a dict containing standard a standard config dictionary for each
    model_artifact_id in model_artifact_ids
    """

    model_artifact_ids = multi_model_test_config['model_artifact_ids']
    common_config = copy.deepcopy(multi_model_test_config)
    del common_config['model_artifact_ids']

    model_test_configs = {}

    for i, model_artifact_id in enumerate(model_artifact_ids):
        model_test_configs[i] = copy.deepcopy(common_config)
        model_test_configs[i]['model_artifact_id'] = model_artifact_id

    return model_test_configs


def run_multi_model_test(model_test_configs):

    for key, run_config in model_test_configs.items():
        test_model(run_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='multi_model_test_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'multi_model_test_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()

    _multi_model_test_config = get_config(args_passed)
    _model_test_configs = extract_model_test_configs(_multi_model_test_config)

    run_multi_model_test(_model_test_configs)
