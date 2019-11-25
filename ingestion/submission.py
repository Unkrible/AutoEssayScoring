import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest

from constant import ESSAY_INDEX, ESSAY_LABEL
from ingestion.metrics import kappa
from models.model import Model
from models.classifiers.lgb_classifier import LgbClassifier
from models.classifiers.elastic_net import ElasticNetClassifier

if __name__ == '__main__':
    start = 1
    stop = 9
    result = []
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
        data = pd.concat([train_data, valid_data])
        label = pd.concat([train_label, valid_label])
        # data = train_data
        # label = train_label
        selectK = SelectKBest(k=30)
        model = Model({}, LgbClassifier)
        data = selectK.fit_transform(data, label[ESSAY_LABEL])
        model.fit((data, label))
        test_preds = model.predict(selectK.transform(test_data))
        tmp = pd.DataFrame({'Essay-ID': test_data.index})
        tmp.set_index('Essay-ID', drop=True, inplace=True)
        tmp['Essay-Set'] = set_id
        tmp['Prediction'] = test_preds
        result.append(tmp)
    result = pd.concat(result)
    result['Prediction'] = result['Prediction'].apply(np.round)
    result.to_csv('MG1933078.tsv', sep='\t', header=False)
    print(result)
