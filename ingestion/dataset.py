"""
  AutoEssayScoring datasets.
"""
import pandas as pd
from collections import namedtuple
from pathlib import Path

from common import log, timeit
from constant import *

Dataset = namedtuple("Dataset", ['data', 'label'])


class AutoEssayScoringDataset:
    """"AutoEssayScoringDataset"""

    def __init__(self, dataset_dir, essay_set_id=None):
        self._dataset_dir = Path(dataset_dir)
        self._essay_set_id = essay_set_id
        self._all_data = None
        self._all_label = None
        self._train_data = None
        self._train_label = None
        self._valid_data = None
        self._valid_label = None
        self._test_data = None
        log(f"Begin read data from {dataset_dir}", DEBUG)
        self._read_all()
        self._meta_info = self._get_meta()

    def _get_meta(self):
        meta_info = {}
        return meta_info

    def _read_all(self):
        train = AutoEssayScoringDataset._read_dataset_by_id(
            self._dataset_dir / 'train.tsv',
            self._essay_set_id
        )
        valid = AutoEssayScoringDataset._read_dataset_by_id(
            self._dataset_dir / 'dev.tsv',
            self._essay_set_id
        )
        self._train_data, self._train_label = AutoEssayScoringDataset._data_label_split(train)
        self._valid_data, self._valid_label = AutoEssayScoringDataset._data_label_split(valid)
        self._test = AutoEssayScoringDataset._read_dataset_by_id(
            self._dataset_dir / 'test.tsv',
            self._essay_set_id,
            use_cols=None
        )
        self._test_data = self._test['essay']

    @staticmethod
    def _read_dataset_by_id(path, set_id, use_cols=USE_COLUMNS):
        log(f"Begin read data from {path}/{'all' if set_id is None else set_id}", DEBUG)
        dataset = AutoEssayScoringDataset._read_dataset(path, use_columns=use_cols)
        dataset = AutoEssayScoringDataset._select_essay_set(dataset, set_id)
        return dataset

    @property
    def all_train(self):
        if self._all_data is None or self._all_label is None:
            self._all_data = pd.concat([self._train_data, self._valid_data])
            self._all_label = pd.concat([self._train_label, self._valid_label])
        assert len(self._all_data) == len(self._all_label)
        return Dataset(self._all_data, self._all_label)

    @property
    def train(self):
        assert len(self._train_data) == len(self._train_label)
        return Dataset(self._train_data, self._train_label)

    @property
    def valid(self):
        assert len(self._valid_data) == len(self._valid_label)
        return Dataset(self._valid_data, self._valid_label)

    @property
    def test(self):
        return Dataset(self._test_data, None)

    @property
    def meta(self):
        return self._meta_info

    @staticmethod
    def _read_dataset(file_path, sep='\t', index=ESSAY_INDEX, use_columns=None):
        dataset = pd.read_csv(
            file_path,
            sep=sep,
            usecols=use_columns
        )
        if index is not None:
            dataset.set_index(index, drop=True, inplace=True)
        return dataset

    @staticmethod
    def _select_essay_set(dataset, set_index):
        if set_index is not None:
            return dataset[dataset['essay_set'] == set_index]
        else:
            return dataset

    @staticmethod
    def _data_label_split(dataset, label=ESSAY_LABEL):
        return dataset['essay'], dataset[label]


@timeit
def get_dataset(args):
    """get dataset"""
    dataset_dir = args.dataset_dir
    dataset_id = args.dataset

    dataset = AutoEssayScoringDataset(dataset_dir, dataset_id)
    return dataset


if __name__ == '__main__':

    dataset = AutoEssayScoringDataset("D:\\essay_data", 1)

    print(dataset.train.data)
    # dataset = AutoEssayScoringDataset("~/Project/AutoEssayScoring/essay_data", 2)
    dataset = AutoEssayScoringDataset("../resources/essay_data", 2)

    tmp = dataset.all_train
