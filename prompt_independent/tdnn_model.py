import numpy as np

from models.classifier import Classifier
from prompt_independent.rank_model import LgbRankModel
from prompt_independent.dnn_model import DNNModel


class TDNNClassifier(Classifier):

    def __init__(self, **kwargs):
        super(TDNNClassifier, self).__init__()
        self._rand_model = None
        self._score_model = None
        
    def fit(self, dataset, *args, **kwargs):
        self._rand_model = LgbRankModel()
        self._rand_model.fit(dataset, *args, **kwargs)

    def predict(self, data, *args, **kwargs):
        # [0, 4] low, [8, 10] high
        y_hat = self._rand_model.predict(data)
        neg_index = y_hat < 4
        pos_index = y_hat >= 8
        x_train = data[neg_index or pos_index]
        y_hat[neg_index] = 0
        y_hat[pos_index] = 1
        y_train = y_hat[neg_index or pos_index]

        self._score_model = DNNModel()
        self._score_model.fit((x_train, y_train))
        return self._score_model.predict(x_train)
