import torchvision.models as models
import torch
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, ORIGINAL_PRETRAINED_MODELS, WANDB_PID, REL_PATHS
from src.d00_utils.functions import get_model_path, save_model, get_config  # ,read_json_artifact, load_wandb_model
from src.d00_utils.classes import Sat6ResNet, Sat6ResNet50, Sat6DenseNet161
import argparse

import wandb
wandb.login()


def load_pretrained_model(model_id):

    """
    Returns pre-trained model. Places365 model loading code pulled from https://github.com/CSAILVision/places365, with
    densenet161 state dict reconfiguration pulled from xeroxM comment at
    https://github.com/CSAILVision/places365/issues/53
    """

    model_metadata = ORIGINAL_PRETRAINED_MODELS[model_id]
    arch = model_metadata['arch']
    model_file_config = model_metadata['model_file_config']
    model_path = get_model_path(**model_file_config)

    # putting each model inside an if statement to ensure correct state dict key replacements
    if model_id == 'resnet18_places365_as_downloaded':
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        return model

    if model_id == 'resnet50_places365_as_downloaded':
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        return model

    if model_id == 'densenet161_places365_as_downloaded':

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k, 'norm.', 'norm'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'conv.', 'conv'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normweight', 'norm.weight'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normrunning', 'norm.running'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normbias', 'norm.bias'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'convweight', 'conv.weight'): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        return model

    if model_id == 'resnet18_sat6':
        model = Sat6ResNet()
        return model

    if model_id == 'resnet50_sat6':
        model = Sat6ResNet50()
        return model

    if model_id == 'densenet161_sat6':
        model = Sat6DenseNet161()
        return model


def load_log_original_model(config):

    model_id = config['model_id']
    description = config['description']
    new_model_id = config['new_model_id']
    new_model_filename = config['new_model_filename']

    model = load_pretrained_model(model_id)

    model_metadata = ORIGINAL_PRETRAINED_MODELS[model_id]
    model_metadata_additions = {
        'description': description,
        'ROOT_DIR_at_run': str(ROOT_DIR),
    }
    model_metadata.update(model_metadata_additions)

    artifact_type = model_metadata['artifact_type']

    # get "new" model dir for model that has been loaded to incorporate the state dict naming update
    new_model_rel_parent_dir = REL_PATHS[artifact_type]
    new_model_rel_dir = Path(new_model_rel_parent_dir, new_model_id)
    new_model_file_config = {
        'model_rel_dir': str(new_model_rel_dir),
        'model_filename': new_model_filename
    }
    model_metadata['model_file_config'] = new_model_file_config  # incorporate correct model_file_config

    # log_metadata(artifact_type, new_model_id, model_metadata)  # removing redundant project metadata
    model_path, helper_path = save_model(model, model_metadata)

    with wandb.init(project=WANDB_PID, job_type='log_model') as run:

        model_artifact = wandb.Artifact(
            new_model_id,
            type=artifact_type,
            metadata=model_metadata,
            description=description
        )

        model_artifact.add_file(model_path)
        model_artifact.add_file(helper_path)
        run.log_artifact(model_artifact)
        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='artifact_log_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'artifact_log_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = get_config(args_passed)

    load_log_original_model(run_config)
