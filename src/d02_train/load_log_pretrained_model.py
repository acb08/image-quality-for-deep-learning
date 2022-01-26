import torchvision.models as models
import torch
from pathlib import Path
from src.d00_utils.definitions import ROOT_DIR, PRETRAINED_MODELS, PROJECT_ID
from src.d00_utils.functions import log_metadata, log_model_helper, read_artifact

import wandb


def load_model(model_path, arch):

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    return model


def load_log_original_model(model_id, description):

    model_config = PRETRAINED_MODELS[model_id]

    model_rel_dir = model_config['model_rel_dir']
    arch = model_config['arch']
    filename = model_config['filename']
    artifact_type = model_config['artifact_type']

    model_dir = Path(ROOT_DIR, model_rel_dir)
    model_path = Path(model_dir, filename)
    model = load_model(model_path, arch)

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

    return model


if __name__ == '__main__':

    _model_id = 'resnet18_pre_trained'
    _description = 'test of loading and logging resnet18 pre-trained'

    _model = load_log_original_model(_model_id, _description)

    import wandb

    run = wandb.init()
    artifact = run.use_artifact('austinbergstrom/places_dry_run/resnet18_pre_trained:latest', type='model')
    artifact_dir = artifact.download()

    helper_data = read_artifact(artifact_dir, 'helper.json')



