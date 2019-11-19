from models.classifier import Classifier
from models.feature_engineering import FeatureEngineer

from common import log, timeit
from models.svm_classifier import SVMClassifier


class Model(Classifier):
    def __init__(self, metadata, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._meta = metadata
        self._fe = FeatureEngineer()
        self._model = SVMClassifier(*args, **kwargs)

    @timeit
    def fit(self, dataset, *args, **kwargs):
        train_data, train_label = dataset
        train_data = self._fe.fit_transform(train_data)
        self._model.fit((train_data, train_label), *args, **kwargs)

    @timeit
    def predict(self, data, *args, **kwargs):
        data = self._fe.transform(data)
        self._model.predict(data, *args, **kwargs)
