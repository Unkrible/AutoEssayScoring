from sklearn.model_selection import train_test_split

from common import timeit
from constant import RANDOM_STATE
from models.classifier import Classifier
from models.classifiers.svm_classifier import SVMClassifier
from models.classifiers.elastic_net import ElasticNetClassifier
from models.classifiers.lgb_classifier import LgbClassifier


class Model(Classifier):
    def __init__(self, metadata, classifier_class, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._meta = metadata
        self._model_class = classifier_class
        self._model = classifier_class(*args, **kwargs)

    @timeit
    def fit(self, dataset, *args, **kwargs):
        train_data, train_label = dataset
        x_hyper, _, y_hyper, _ = train_test_split(
            train_data, train_label,
            test_size=0.2, random_state=RANDOM_STATE
        )
        hyper_params = self._model_class.hyper_params_search(
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
    import pandas as pd
    import numpy as np
    from constant import ESSAY_INDEX, ESSAY_LABEL
    from ingestion.metrics import kappa
    from sklearn.feature_selection import SelectKBest
    original_kappas = [
        0.8026615360220601,
        0.6453843679639755,
        0.6675004632203076,
        0.6201412784903135,
        0.7983501888241695,
        0.6810744597077609,
        0.733541110264561,
        0.7055995750817732
    ]
    start = 1
    stop = 9
    my_kappas = []
    weights = np.ones(stop - start, dtype=np.float)
    y_hat = []
    for set_id in range(start, stop):
        csv_params = {
            'index_col': ESSAY_INDEX,
            'dtype': {'domain1_score': np.float}
        }
        train_data = pd.read_csv(f"../features/TrainSet{set_id}.csv", **csv_params)
        train_label = pd.read_csv(f"../features/TrainLabel{set_id}.csv", **csv_params)
        valid_data = pd.read_csv(f"../features/ValidSet{set_id}.csv", **csv_params)
        valid_label = pd.read_csv(f"../features/ValidLabel{set_id}.csv", **csv_params)
        test_data = pd.read_csv(f"../features/TestSet{set_id}.csv", **csv_params)
        test_label = pd.read_csv(f"../features/TestLabel{set_id}.csv", **csv_params)
        selectK = SelectKBest(k=30)
        data = pd.concat([train_data, valid_data])
        label = pd.concat([train_label, valid_label])
        # data = train_data
        # label = train_label
        data = selectK.fit_transform(data, label[ESSAY_LABEL])
        model = Model({}, LgbClassifier)
        model.fit((data, label))
        y_preds = model.predict(selectK.transform(test_data))
        y_hat.append(y_preds)
        res = kappa(test_label[ESSAY_LABEL].tolist(), y_preds)
        my_kappas.append(res)
        # weights[set_id - start] = len(valid_label)
    df = pd.DataFrame({'Baseline': original_kappas[start - 1: stop - 1], "Our's": my_kappas})
    df['Improvement'] = df["Our's"] - df['Baseline']
    weights = weights / np.sum(weights)
    df['Set Weight'] = weights
    print(f"Final result:\n{df}")
    print(f"{np.average(my_kappas, weights=weights)}")
