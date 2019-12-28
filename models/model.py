from sklearn.model_selection import train_test_split

from common import timeit
from constant import RANDOM_STATE
from ingestion.metrics import kappa
from models.classifier import Classifier
from models.classifiers.svm_classifier import SVMClassifier
from models.classifiers.elastic_net import ElasticNetClassifier
from models.classifiers.lgb_classifier import LgbClassifier
from models.classifiers.dense_classifier import DenseClassifier


class Model(Classifier):
    def __init__(self, metadata, classifier_class, *args, hyper_search=True, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._meta = metadata
        self._model_class = classifier_class
        self._model = classifier_class(*args, **kwargs)
        self._hyper_search = hyper_search

    @timeit
    def fit(self, dataset, *args, **kwargs):
        train_data, train_label = dataset
        x_hyper, _, y_hyper, _ = train_test_split(
            train_data, train_label,
            test_size=0.2, random_state=RANDOM_STATE
        )
        if self._hyper_search:
            hyper_params = self._model_class.hyper_params_search(
                (x_hyper, y_hyper),
                *args,
                max_evals=50,
                **kwargs
            )
            self._model.fit((train_data, train_label), *args, **hyper_params)
        else:
            self._model.fit((train_data, train_label), *args)

    @timeit
    def predict(self, data, *args, **kwargs):
        return self._model.predict(data, *args, **kwargs)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from constant import ESSAY_INDEX, ESSAY_LABEL

    feature_set = "hisk"
    start = 1
    stop = 2
    results = []
    for set_id in range(start, stop):
        csv_params = {
            'index_col': ESSAY_INDEX,
            'dtype': {'domain1_score': np.float}
        }
        train_data_sets = []
        train_label_sets = []
        valid_data_sets = []
        valid_label_sets = []
        for each in range(1, 9):
            if each == set_id:
                continue
            train_data = pd.read_csv(f"../{feature_set}/TrainSet{set_id}.csv", **csv_params)
            train_label = pd.read_csv(f"../{feature_set}/TrainLabel{set_id}.csv", **csv_params)
            valid_data = pd.read_csv(f"../{feature_set}/ValidSet{set_id}.csv", **csv_params)
            valid_label = pd.read_csv(f"../{feature_set}/ValidLabel{set_id}.csv", **csv_params)
            train_data_sets.append(train_data)
            train_label_sets.append(train_label)
            valid_data_sets.append(valid_data)
            valid_label_sets.append(valid_label)
        train_data = pd.concat(train_data_sets)
        train_label = pd.concat(train_label_sets)
        valid_data = pd.concat(valid_data_sets)
        valid_label = pd.concat(valid_label_sets)
        test_data = pd.read_csv(f"../{feature_set}/TestSet{set_id}.csv", **csv_params)
        test_label = pd.read_csv(f"../{feature_set}/TestLabel{set_id}.csv", **csv_params)

        data: pd.DataFrame = pd.concat([train_data, valid_data])
        label = pd.concat([train_label, valid_label])
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        model = Model({}, LgbClassifier)
        model.fit((data, label[ESSAY_LABEL]))
        test_data = scaler.transform(test_data)
        y_hat = model.predict(test_data)
        res = kappa(test_label[ESSAY_LABEL], y_hat)
        results.append(res)
        print(f"{set_id}: {feature_set} kappa {res}")
    print(results)
    print(np.mean(results))
