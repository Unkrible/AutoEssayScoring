from sklearn.model_selection import train_test_split

from common import timeit
from constant import RANDOM_STATE
from models.classifier import Classifier
from models.feature_engineering import FeatureEngineer
from models.classifiers.svm_classifier import SVMClassifier
from models.classifiers.elastic_net import ElasticNetClassifier


class Model(Classifier):
    def __init__(self, metadata, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._meta = metadata
        self._fe = FeatureEngineer()
        self._model = ElasticNetClassifier(*args, **kwargs)

    @timeit
    def fit(self, dataset, *args, **kwargs):
        train_data, train_label = dataset
        train_data = self._fe.fit_transform(train_data)
        x_hyper, _, y_hyper, _ = train_test_split(
            train_data, train_label,
            test_size=0.7, random_state=RANDOM_STATE
        )
        hyper_params = ElasticNetClassifier.hyper_params_search(
            (x_hyper, y_hyper),
            *args,
            max_evals=50,
            **kwargs
        )
        self._model.fit((train_data, train_label), *args, **hyper_params)

    @timeit
    def predict(self, data, *args, **kwargs):
        data = self._fe.transform(data)
        return self._model.predict(data, *args, **kwargs)
