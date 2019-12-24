import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from constant import ESSAY_INDEX, ESSAY_LABEL, RANK
from prompt_independent.rank_model import LgbRankModel


def read_datasets(path, set_id, read, **csv_params):
    data_ids = [i for i in range(1, 9) if i != set_id]
    data = []
    for id in data_ids:
        data.append(read(path, id, **csv_params))
    data = pd.concat(data)
    return data


def read_label(path, id, **csv_params):
    label = read_dataset(path, id, **csv_params)
    scaler = MinMaxScaler(feature_range=(0, 10))
    label_scaled = scaler.fit_transform(np.asarray(label[ESSAY_LABEL]).reshape((-1, 1)))
    label[ESSAY_LABEL] = label_scaled
    return label[ESSAY_LABEL].astype(np.int)


def read_dataset(path, id, **csv_params):
    return pd.read_csv(f"{path}{id}.csv", **csv_params)


if __name__ == '__main__':
    set_id = 1
    feature_file = "prompt-independent/dataframes3"
    csv_params = {
        'index_col': ESSAY_INDEX,
        'dtype': {'domain1_score': np.float}
    }

    # read data
    train_data = read_datasets(f"../{feature_file}/TrainSet", set_id, read_dataset, **csv_params)
    train_label = read_datasets(f"../{feature_file}/TrainLabel", set_id, read_label, **csv_params)
    valid_data = read_datasets(f"../{feature_file}/ValidSet", set_id, read_dataset, **csv_params)
    valid_label = read_datasets(f"../{feature_file}/ValidLabel", set_id, read_label, **csv_params)
    test_data = read_dataset(f"../{feature_file}/TestSet", set_id, **csv_params)
    test_label = read_label(f"../{feature_file}/TestLabel", set_id, **csv_params)

    # train model
    rank_model = LgbRankModel()
    rank_model.fit((train_data, train_label))
    rank_hat = rank_model.predict(test_data)
    test_data[RANK] = rank_hat
    sort_index = test_data[RANK].sort_values().index
    bad_index = sort_index[:int(len(sort_index) * 0.4)]
    good_index = sort_index[int(len(sort_index) * 0.7):]
    print("here")
