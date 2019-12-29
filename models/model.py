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

    asap_ranges = {
        1: (2.0, 12.0),
        2: (1.0, 6.0),
        3: (0.0, 3.0),
        4: (0.0, 3.0),
        5: (0.0, 4.0),
        6: (0.0, 4.0),
        7: (0.0, 30.0),
        8: (0.0, 60.0)
    }

    # 12  78  134  56
    sets = {
        1: [2],
        2: [1],
        3: [1, 4],
        4: [1, 3],
        5: [6],
        6: [5],
        7: range(1, 9),
        8: range(1, 9)
    }

    feature_set = "dataframes4"
    start = 1
    stop = 9
    outputs = []
    for set_id in range(start, stop):
        csv_params = {
            'index_col': ESSAY_INDEX,
            'dtype': {'domain1_score': np.float}
        }
        train_data_sets = []
        train_label_sets = []
        valid_data_sets = []
        valid_label_sets = []
        for each in sets[set_id]:
            if each == set_id:
                continue
            label_scaler = MinMaxScaler(feature_range=(0, 10))
            train_data = pd.read_csv(f"../{feature_set}/TrainSet{each}.csv", **csv_params)
            train_label = pd.read_csv(f"../{feature_set}/TrainLabel{each}.csv", **csv_params)
            train_label = pd.Series(
                label_scaler.fit_transform(
                    train_label[ESSAY_LABEL].values.reshape(-1, 1)
                ).reshape(-1),
                index=train_label.index
            )
            valid_data = pd.read_csv(f"../{feature_set}/ValidSet{each}.csv", **csv_params)
            valid_label = pd.read_csv(f"../{feature_set}/ValidLabel{each}.csv", **csv_params)
            valid_label = pd.Series(
                label_scaler.transform(valid_label[ESSAY_LABEL].values.reshape(-1, 1)).reshape(-1),
                index=valid_label.index
            )

            train_data_sets.append(train_data)
            train_label_sets.append(train_label)
            valid_data_sets.append(valid_data)
            valid_label_sets.append(valid_label)
        train_data = pd.concat(train_data_sets)
        train_label = pd.concat(train_label_sets)
        valid_data = pd.concat(valid_data_sets)
        valid_label = pd.concat(valid_label_sets)
        test_data = pd.read_csv(f"../{feature_set}/TestSet{set_id}.csv", **csv_params)
        index = test_data.index
        data: pd.DataFrame = pd.concat([train_data, valid_data])
        label = pd.concat([train_label, valid_label])
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        model = Model({}, LgbClassifier)
        model.fit((data, label))
        test_data = scaler.transform(test_data)
        y_hat = model.predict(test_data)
        label_scaler = MinMaxScaler(feature_range=asap_ranges[set_id])
        y_hat = label_scaler.fit_transform(y_hat.reshape(-1, 1)).reshape(-1)
        tmp = pd.DataFrame({ESSAY_INDEX: index})
        tmp.set_index(ESSAY_INDEX, drop=True, inplace=True)
        tmp['essay_set'] = set_id
        tmp['pred'] = y_hat
        outputs.append(tmp)
    result = pd.concat(outputs)
    result['pred'] = result['pred'].apply(np.round)
    result.to_csv('MG1933078.tsv', sep='\t', header=False)
