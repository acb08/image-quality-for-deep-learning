import src.d02_train.train_detection
from src.d02_train import train_detection as run
import json
import torch
import src.d00_utils.definitions as definitions
from pathlib import Path

import src.d03_test.eval_detection_model
import src.d03_test.test_model
from src.d00_utils.detection_functions import listify


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

    outputs, targets = src.d02_train.train_detection.evaluate(model=model,
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
    _output_dir = f'test_result_check2_{_cutoff}-img'
    main(cutoff=_cutoff,
         output_dir=_output_dir)
