from common import timeit
from models.classifier import Classifier
from models.classifiers.elastic_net import ElasticNetClassifier
from models.classifiers.lgb_classifier import LgbClassifier


class EnsembleModel(Classifier):
    def __init__(self, *args, candidates=None, **kwargs):
        super(EnsembleModel, self).__init__()
        if candidates is None:
            candidates = [ElasticNetClassifier, LgbClassifier]
        self._submodels = candidates
