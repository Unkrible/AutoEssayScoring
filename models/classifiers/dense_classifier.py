import tensorflow as tf
from hyperopt import hp
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.metrics import mean_squared_error
from itertools import combinations

from ingestion.metrics import kappa
from models.classifier import Classifier
from models.hyper_opt import hyper_opt


def _dense_model(*args, units=None, loss='mean_squared_error', **kwargs):
    clear_session()
    if units is None:
        units = [100]

    model = Sequential()
    for unit in units:
        model.add(Dense(unit))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = Adam()
    model.compile(optimizer, loss=loss, metrics=['mse', 'mae'])
    return model


class DenseClassifier(Classifier):

    def __init__(self, *args, **kwargs):
        super(DenseClassifier, self).__init__()
        self._model_args = args
        self._model_kwargs = kwargs
        self._model = None

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        self._model = _dense_model(
            *[*self._model_args, *args],
            **{**self._model_kwargs, **kwargs}
        )
        self._model.fit(x, y.values, shuffle=True)

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data)

    units_candidates = [2000, 1000, 500, 100, 50, 30, 10]
    units_combinations = [
        *list(combinations(units_candidates, 2)),
        *list(combinations(units_candidates, 3)),
        *list(combinations(units_candidates, 4)),
        units_candidates,
        []
    ]
    hyper_space = {
        'loss': hp.choice("loss", [tf.losses.mean_squared_error, tf.losses.huber_loss]),
        'units': hp.choice("units", units_combinations)
    }

    @staticmethod
    def hyper_params_search(dataset, *args, **kwargs):
        x, y = dataset
        hyper_params = hyper_opt(x, y, {}, DenseClassifier.hyper_space, DenseClassifier, kappa, max_evals=40)
        return hyper_params
