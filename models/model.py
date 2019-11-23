from sklearn.model_selection import train_test_split

from common import timeit
from constant import RANDOM_STATE
from models.classifier import Classifier
from models.classifiers.svm_classifier import SVMClassifier
from models.classifiers.elastic_net import ElasticNetClassifier


class Model(Classifier):
    def __init__(self, metadata, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._meta = metadata
        self._model = ElasticNetClassifier(*args, **kwargs)

    @timeit
    def fit(self, dataset, *args, **kwargs):
        train_data, train_label = dataset
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
        return self._model.predict(data, *args, **kwargs)


if __name__ == '__main__':
    import pickle
    from collections import namedtuple
    FeatureDatasets = namedtuple("FeatureDatasets", ['train', 'train_label', 'valid', 'valid_label', 'test'])
    set_id = 1
    with open(f'../features/SyntaxFeatureLabelSet{set_id}.pl', 'rb') as f:
        dataset = pickle.load(f)
    print('here')
