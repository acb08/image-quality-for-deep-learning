import run_model as run
import json
import torch
import src.d00_utils.definitions as definitions
from pathlib import Path


def listify(torch_data_dict):

    listed_data_dict = {}

    for image_id, torch_image_data in torch_data_dict.items():

        image_data = {}

        for key, torch_data in torch_image_data.items():
            if type(torch_data) == torch.Tensor:
                list_data = torch_data.tolist()
            else:
                list_data = torch_data
            image_data[key] = list_data

        listed_data_dict[image_id] = image_data

    return listed_data_dict


def main(cutoff=None, batch_size=2, output_dir='test_result', output_filename='result.json'):

    dataset = run.get_dataset(cutoff=cutoff)
    loader = run.get_loader(dataset=dataset,
                            batch_size=batch_size)
    model = run.get_model()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    outputs, targets = run.evaluate(model=model,
                                    data_loader=loader,
                                    device=device)

    outputs = listify(outputs)
    targets = listify(targets)

    result = {'outputs': outputs, 'targets': targets}

    output_dir = Path(definitions.ROOT_DIR, output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(exist_ok=True, parents=True)

    with open(Path(output_dir, output_filename), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':

    _cutoff = 4
    _output_dir = f'test_result_{_cutoff}-img'
    main(cutoff=_cutoff,
         output_dir=_output_dir)
