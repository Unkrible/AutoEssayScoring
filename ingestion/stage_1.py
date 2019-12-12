from models.model import Model
from models.classifiers.lgb_classifier import LgbClassifier
from models.classifiers.elastic_net import ElasticNetClassifier


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    from constant import ESSAY_INDEX, ESSAY_LABEL

    feature_sets = ["hisk", "hisk-15", "features"]
    start = 1
    stop = 9
    result = []
    final_choose = [None, [4], [1, 4], [2, 3, 4], [3], [4], [1, 3], [1], [2, 4]]
    for set_id in range(start, stop):
        csv_params = {
            'index_col': ESSAY_INDEX,
            'dtype': {'domain1_score': np.float}
        }
        y_preds = []
        for feature_set in feature_sets:
            train_data = pd.read_csv(f"../{feature_set}/TrainSet{set_id}.csv", **csv_params)
            train_label = pd.read_csv(f"../{feature_set}/TrainLabel{set_id}.csv", **csv_params)
            valid_data = pd.read_csv(f"../{feature_set}/ValidSet{set_id}.csv", **csv_params)
            valid_label = pd.read_csv(f"../{feature_set}/ValidLabel{set_id}.csv", **csv_params)
            test_data = pd.read_csv(f"../{feature_set}/TestSet{set_id}.csv", **csv_params)

            data = pd.concat([train_data, valid_data])
            label = pd.concat([train_label, valid_label])

            model1 = Model({}, LgbClassifier)
            model1.fit((data, label[ESSAY_LABEL]))
            y_preds.append(model1.predict(test_data))
            model2 = Model({}, ElasticNetClassifier, hyper_search=False)
            model2.fit((data, label[ESSAY_LABEL]))
            y_preds.append(model2.predict(test_data))
        y_hat = np.average([y_preds[i] for i in final_choose[set_id]], axis=0)

        tmp = pd.DataFrame({ESSAY_INDEX: test_data.index})
        tmp.set_index(ESSAY_INDEX, drop=True, inplace=True)
        tmp['essay_set'] = set_id
        tmp['pred'] = y_hat
        result.append(tmp)
        # weights[set_id - start] = len(valid_label)
    result = pd.concat(result)
    result['pred'] = result['pred'].apply(np.round)
    result.to_csv('MG1933078.tsv', sep='\t', header=False)
    print(final_choose)