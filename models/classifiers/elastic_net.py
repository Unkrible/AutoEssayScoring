from sklearn.linear_model import ElasticNet
from hyperopt import hp

from constant import RANDOM_STATE
from ingestion.metrics import kappa
from models.classifier import Classifier
from models.hyper_opt import hyper_opt


class ElasticNetClassifier(Classifier):

    def __init__(self, *args, **kwargs):
        super(ElasticNetClassifier, self).__init__()
        self._model = ElasticNet(*args, **kwargs, max_iter=10000, random_state=RANDOM_STATE)

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        self._model.fit(x, y)

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data)

    @staticmethod
    def hyper_params_search(dataset, *args, **kwargs):
        x, y = dataset
        hyper_space = {
            'l1_ratio': hp.uniform('l1_ratio', .01, .9),
            'alpha': hp.uniform('alpha', .01, 1)
        }
        hyper_params = hyper_opt(x, y, {}, hyper_space, ElasticNetClassifier, kappa)
        return hyper_params
