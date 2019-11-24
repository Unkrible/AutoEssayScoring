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
    import pandas as pd
    import numpy as np
    from constant import ESSAY_INDEX, ESSAY_LABEL
    from ingestion.metrics import kappa
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
    stop = 2
    my_kappas = []
    for set_id in range(start, stop):
        csv_params = {
            'index_col': ESSAY_INDEX,
            'dtype': {'domain1_score': np.float}
        }
        train_data = pd.read_csv(f"../features/SyntaxFeatureLabelTrainSet{set_id}.csv", **csv_params)
        train_label = pd.read_csv(f"../features/SyntaxFeatureLabelTrainLabel{set_id}.csv", **csv_params)
        valid_data = pd.read_csv(f"../features/SyntaxFeatureLabelValidSet{set_id}.csv", **csv_params)
        valid_label = pd.read_csv(f"../features/SyntaxFeatureLabelValidLabel{set_id}.csv", **csv_params)
        test_data = pd.read_csv(f"../features/SyntaxFeatureLabelTestSet{set_id}.csv", **csv_params)
        model = Model({})
        model.fit((train_data, train_label))
        y_preds = model.predict(valid_data)
        res = kappa(valid_label[ESSAY_LABEL].tolist(), y_preds)
        my_kappas.append(res)
    df = pd.DataFrame({'Baseline': original_kappas[start - 1: stop - 1], 'Our solution': my_kappas})
    df['Improvement'] = df['Our solution'] - df['Baseline']
    print(f"Final result:\n{df}")
