import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constant import RANDOM_STATE
from models.classifier import Classifier


# TODO: 特征工程
class LgbRankModel(Classifier):

    def __init__(self, **kwargs):
        super(LgbRankModel, self).__init__()
        self._model_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "num_threads": 5
        }
        self._model_hyper_params = kwargs
        self._model = None
        self._cut_bins = None

    def preprocess_dataset(self, x, y):
        index = np.argsort(y)
        x, y = x[index, :], y[index]
        group = np.histogram(y, bins=self._cut_bins)
        return x, y, group

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        _, self._cut_bins = pd.qcut(y, 10)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y,
            test_size=0.2, random_state=RANDOM_STATE
        )
        x_train, y_train, group_train = self.preprocess_dataset(x_train, y_train)
        x_valid, y_valid, group_valid = self.preprocess_dataset(x_valid, y_valid)
        train_data = lgb.Dataset(x_train, label=y_train, group=group_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, group=group_valid)
        self._model = lgb.train(
            {**self._model_params, **self._model_hyper_params},
            train_data,
            300,
            valid_data,
            early_stopping_rounds=30,
            verbose_eval=0
        )

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data)
