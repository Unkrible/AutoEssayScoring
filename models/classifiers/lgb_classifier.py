from hyperopt import hp
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest

from constant import RANDOM_STATE
from ingestion.metrics import kappa
from models.classifier import Classifier
from models.hyper_opt import hyper_opt


class LgbClassifier(Classifier):

    def __init__(self, *args, select_feature_num=30, **kwargs):
        super(LgbClassifier, self).__init__()
        self._model_params = {
            "objective": "regression_l2",
            "metric": "l2",
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "num_threads": 5
        }
        self._model_hyper_params = kwargs
        self._model = None
        self._selectK = SelectKBest(k=select_feature_num)

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        x = self._selectK.fit_transform(x, y)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y,
            test_size=0.2, random_state=RANDOM_STATE
        )
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid)
        self._model = lgb.train(
            {**self._model_params, **self._model_hyper_params}, train_data, 300, valid_data,
            early_stopping_rounds=30, verbose_eval=0
        )

    def predict(self, data, *args, **kwargs):
        return self._model.predict(self._selectK.transform(data))

    @staticmethod
    def hyper_params_search(dataset, *args, **kwargs):
        x, y = dataset
        hyper_space = {
            "select_feature_num": hp.choice("select_feature_num", np.linspace(15, 44, 15, dtype=int)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
            "max_depth": hp.choice("max_depth", [-1, 5, 6, 7]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.6, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10)
        }
        hyper_params = hyper_opt(x, y, {}, hyper_space, LgbClassifier, kappa, max_evals=100)
        return hyper_params
