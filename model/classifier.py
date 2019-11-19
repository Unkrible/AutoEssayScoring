import abc


class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, dataset, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data, *args, **kwargs):
        pass


class Model(Classifier):
    def __init__(self, metadata, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._meta = metadata

    def fit(self, dataset, *args, **kwargs):
        pass

    def predict(self, data, *args, **kwargs):
        pass
