import torchvision.models as models
import torch
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, ORIGINAL_PRETRAINED_MODELS, PROJECT_ID
from src.d00_utils.functions import log_metadata, log_model_helper, read_artifact

import wandb


def load_pretrained_model(model_id):

    """
    Returns pre-trained model
    """

    model_metadata = ORIGINAL_PRETRAINED_MODELS[model_id]
    arch = model_metadata['arch']
    model_file_config = model_metadata['model_file_config']
    model_path = get_model_path(**model_file_config)

    # putting each model inside of an if statement to ensure correct state dict key replacements
    if model_id == 'resnet18_places365':
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

    return model


# TODO: consider moving get_model_path() to functions.py
def get_model_path(model_rel_dir, model_filename):
    """
    Arguments named so function can be called with to be called with **model_file_config
    """
    model_path = Path(ROOT_DIR, model_rel_dir, model_filename)
    return model_path


def load_log_original_model(model_id, description):

    model_config = ORIGINAL_PRETRAINED_MODELS[model_id]

    model_rel_dir = model_config['model_rel_dir']
    filename = model_config['filename']
    artifact_type = model_config['artifact_type']

    model_dir = Path(ROOT_DIR, model_rel_dir)
    model_path = Path(model_dir, filename)
    model = load_pretrained_model(model_id)

    model_metadata = {
        'description': description,
        'artifact_filename': filename,
        'ROOT_DIR_at_run': str(ROOT_DIR),
        'artifact_type': 'model'
    }
    model_metadata.update(model_config)
    log_metadata(artifact_type, model_id, model_metadata)
    helper_path = log_model_helper(model_dir, model_metadata)

    with wandb.init(project=PROJECT_ID, job_type='log_model') as run:

        model_artifact = wandb.Artifact(
            model_id,
            type=artifact_type,
            metadata=model_metadata,
            description=description
        )

        model_artifact.add_file(model_path)
        model_artifact.add_file(helper_path)
        run.log_artifact(model_artifact)
        wandb.finish()


if __name__ == '__main__':

    _model_id = 'resnet18_places365'
    _description = 'test of loading and logging resnet18 pre-trained'

    load_log_original_model(_model_id, _description)

    run = wandb.init()
    artifact = run.use_artifact('austinbergstrom/places_dry_run/resnet18_pre_trained:latest', type='model')
    artifact_dir = artifact.download()

    helper_data = read_artifact(artifact_dir, 'helper.json')



