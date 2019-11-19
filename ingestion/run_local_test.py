import argparse
from pathlib import Path
from common import log, _here
from ingestion.dataset import get_dataset
from ingestion.metrics import kappa
from model.classifier import Model


def _parse_args():
    default_starting_kit_dir = Path(_here())
    default_dataset_dir = default_starting_kit_dir / 'essay_data'
    default_code_dir = default_starting_kit_dir / 'model'
    default_time_budget = 1200

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset, should contain"
                             "'data' and 'solution'")

    parser.add_argument('--dataset', type=int,
                        default=None,
                        help="Which essay set will be use, None means all essay sets")

    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")

    parser.add_argument("--time_budget", type=float,
                        default=default_time_budget,
                        help="Time budget for training model if not specified"
                             " in info.json.")

    parser.add_argument("--test", type=bool,
                        default=True,
                        help="If test, use train data to train and predict valid data,"
                             "else, use all data to train and predict test")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    dataset = get_dataset(args)
    model = Model(dataset.meta)

    is_test = args.test
    if is_test:
        train_data = dataset.train
    else:
        train_data = dataset.all_train
    model.fit(train_data)

    if is_test:
        valid_data, y_true = dataset.valid
        y_pred = model.predict(valid_data)
        res = kappa(y_true, y_pred, weights='quadratic')
        log(f"Kappa is {res} on dataset {args.dataset}")
    else:
        model.predict(dataset.test)
