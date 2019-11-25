from sklearn.svm import LinearSVC

from models.classifier import Classifier


class SVMClassifier(Classifier):
    def __init__(self, *args, **kwargs):
        super(SVMClassifier, self).__init__()
        self._model = LinearSVC()

    def fit(self, dataset, *args, **kwargs):
        train_data, train_label = dataset
        self._model.fit(train_data, train_label)

    def predict(self, data, *args, **kwargs):
        return self._model.decision_function(data)
