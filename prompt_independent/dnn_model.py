import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, \
    Convolution1D, GlobalMaxPooling1D, merge, Dropout, BatchNormalization, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers

from models.classifier import Classifier


class DNNModel(Classifier):
    def __init__(self, hidden_dim, dense_dim,  embedding_dim, embedding_table, max_words, opt="adam", **kwargs):
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim

        self.embedding_dim = embedding_dim
        self.embedding_table = embedding_table
        self.vocab_size = len(embedding_table)

        self.filters = 1
        self.filter_len = 1

        self.max_sens = 10
        self.max_words = max_words
        self.nb_feature = 21

        self.opt = opt
        self.model = self.model_essay()
        super(DNNModel, self).__init__()

    # uses only the semantic network
    def model_essay(self):
        input_words = Input(shape=(self.max_words,), dtype='int32')
        embedding_layer = Embedding(input_dim=self.vocab_size,
                                    output_dim=self.embedding_dim,
                                    weights=self.embedding_table,
                                    trainable=False,
                                    mask_zero=True)(input_words)
        bi_lstm_layer = Bidirectional(
            LSTM(output_dim=self.hidden_dim, return_sequences=False),
            merge_mode='concat'
        )(embedding_layer)

        sentence_model = Model(inputs=input_words, outputs=bi_lstm_layer)

        input_essay = Input(shape=(None, self.max_words), dtype='int32')

        essay_layer = TimeDistributed(sentence_model)(input_essay)

        essay_bilstm_layer = Bidirectional(
            LSTM(output_dim=self.hidden_dim, return_sequences=False),
            merge_mode='concat'
        )(essay_layer)

        bn_merge_layer2 = BatchNormalization()(essay_bilstm_layer)
        merge_dense_layer2 = Dense(self.dense_dim, activation='relu')(bn_merge_layer2)
        score_layer = Dense(1, activation='sigmoid', name='pred_score')(merge_dense_layer2)
        essay_model = Model(inputs=input_essay, outputs=score_layer)

        if self.opt == "adam":
            optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=0, clipvalue=10)
        elif self.opt == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=0, clipvalue=10)
        else:
            optimizer = optimizers.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=0, clipvalue=10)

        essay_model.compile(optimizer=optimizer,
                            loss="mean_squared_error",
                            metrics=['mean_squared_error'])
        return essay_model

    def fit(self, dataset, *args, bad_index=None, good_index=None, **kwargs):
        bad_labels = pd.Series(np.zeros(len(bad_index)), index=bad_index)
        good_labels = pd.Series(np.ones(len(good_index)), index=good_index)
        labels = pd.concat([bad_labels, good_labels])
        data = pd.concat([dataset[bad_index], dataset[good_index]])
        self.model.fit(data, labels)

    def predict(self, data, *args, **kwargs):
        pass
