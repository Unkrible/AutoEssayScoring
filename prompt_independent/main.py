import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence

from constant import ESSAY_INDEX, ESSAY_LABEL, RANK
from prompt_independent.rank_model import LgbRankModel
from prompt_independent.dnn_model import DNNModel


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
    if not (isinstance(id, str) and id.endswith("tsv")):
        id = f"{id}.csv"
    return pd.read_csv(f"{path}{id}", **csv_params)


def read_glove():
    # loading pretrained embedding
    # FT_DIR = '/app/embedding'
    GLOVE_FILE = "/home/chengfeng/glove.6B/glove.6B.300d.txt"
    word_embeddings = {}
    embedding_dim = 0

    i = 0
    # Match the word of each line with the rest words -> fasttext_embeddings_index
    with open(GLOVE_FILE, 'r') as f:
        for line in f:
            if i > 10:
                break
            else:
                i += 1
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_dim = len(coefs)
            word_embeddings[word] = coefs
    print('Found %s glove word vectors.' % len(word_embeddings))
    return word_embeddings, embedding_dim


def sequentialize_data(train_contents):
    MAX_VOCAB_SIZE = 200000
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(train_contents)
    x_train = tokenizer.texts_to_sequences(train_contents)
    max_length = len(max(x_train, key=len))
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    word_index = tokenizer.word_index
    num_features = min(len(word_index) + 1, MAX_VOCAB_SIZE)
    return x_train, word_index, num_features, tokenizer, max_length


def lookup_embedding(word_index, embeddings_index, embedding_dim, num_features):
    embedding_matrix = np.zeros((num_features, embedding_dim))

    beyond_cnt = 0
    hit_cnt = 0
    oov_cnt = 0
    missing = []
    # beyond_num_features = []

    for word, i in word_index.items():
        if i >= num_features:
            beyond_cnt += 1
            # beyond_num_features.append((word, i))
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hit_cnt += 1
        else:
            embedding_matrix[i] = np.random.uniform(-1, 1, size=embedding_dim)
            missing.append(i)
            oov_cnt += 1
    return embedding_matrix


if __name__ == '__main__':
    set_id = 1
    feature_file = "prompt-independent/dataframes3"
    csv_params = {
        'index_col': ESSAY_INDEX,
        'dtype': {'domain1_score': np.float}
    }
    feature_range = [

    ]

    # read data
    train_data = read_datasets(f"../{feature_file}/TrainSet", set_id, read_dataset, **csv_params)
    train_label = read_datasets(f"../{feature_file}/TrainLabel", set_id, read_label, **csv_params)
    valid_data = read_datasets(f"../{feature_file}/ValidSet", set_id, read_dataset, **csv_params)
    valid_label = read_datasets(f"../{feature_file}/ValidLabel", set_id, read_label, **csv_params)
    test_data = read_dataset(f"../{feature_file}/TestSet", set_id, **csv_params)
    test_label = read_label(f"../{feature_file}/TestLabel", set_id, **csv_params)

    # train rank model
    rank_model = LgbRankModel()
    rank_model.fit((train_data, train_label))
    rank_hat = rank_model.predict(test_data)

    labels = pd.Series(rank_hat, index=test_data.index)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 10))
    # test_data[RANK] = rank_hat
    # sort_index = test_data[RANK].sort_values().index
    # bad_index = sort_index[:int(len(sort_index) * 0.4)]
    # good_index = sort_index[int(len(sort_index) * 0.7):]
    #
    # bad_labels = pd.Series(np.zeros(len(bad_index)), index=bad_index)
    # good_labels = pd.Series(np.ones(len(good_index)), index=good_index)
    # labels = pd.concat([bad_labels, good_labels])
    #
    # # preprocess test data
    # csv_params['sep'] = '\t'
    # test_data = read_dataset(f"../essay_data/", "test.tsv", **csv_params)
    # test_data = test_data[test_data["essay_set"] == set_id]["essay"]
    # # train dnn model
    # weights, embedding_dim = read_glove()
    # x_all, word_index, num_features, tokenizer, max_length = sequentialize_data(test_data)
    # embedding_table = lookup_embedding(word_index, weights, embedding_dim, num_features)
    # dnn_model = DNNModel(
    #     hidden_dim=256,
    #     dense_dim=128,
    #     embedding_table=embedding_table,
    #     embedding_dim=embedding_dim,
    #     num_features=num_features,
    #     max_words=max_length
    # )
    # dnn_model.fit(pd.DataFrame(x_all, index=test_data.index), bad_index=bad_index, good_index=good_index)
    # y_all_hat = dnn_model.predict(x_all)
    # print("here")
