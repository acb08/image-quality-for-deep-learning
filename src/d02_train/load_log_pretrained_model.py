import torchvision.models as models
import torch
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, ORIGINAL_PRETRAINED_MODELS, PROJECT_ID, REL_PATHS
from src.d00_utils.functions import log_metadata, get_model_path, save_model, read_json_artifact, load_wandb_model

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
    if model_id == 'resnet18_places365_as_downloaded':
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        return model


# TODO: consider moving get_model_path() to functions.py


def load_log_original_model(model_id, new_model_id, new_model_filename, description):

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

    log_metadata(artifact_type, new_model_id, model_metadata)
    model_path, helper_path = save_model(model, model_metadata)

    with wandb.init(project=PROJECT_ID, job_type='log_model') as run:

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

    _model_id = 'resnet18_places365_as_downloaded'
    _new_model_id = 'resnet18_pretrained_copy'
    _new_model_filename = 'resnet18.pt'
    _description = 'test of loading and logging resnet18 pre-trained'

    load_log_original_model(_model_id, _new_model_id, _new_model_filename, _description)

    run = wandb.init()
    artifact = run.use_artifact('austinbergstrom/places_dry_run/resnet18_pretrained_copy:v0', type='model')
    artifact_rel_dir = artifact.download()
    artifact_dir = Path(Path.cwd(), artifact_rel_dir)

    helper_data = read_json_artifact(artifact_dir, 'helper.json')
    model_filename = helper_data['model_file_config']['model_filename']

    model = load_wandb_model(artifact_dir)






