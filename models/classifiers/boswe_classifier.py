from hyperopt import hp
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Flatten, Dropout, Dense
from tensorflow.python.keras.optimizers import Adam

from ingestion.metrics import kappa
from models.classifier import Classifier
from models.hyper_opt import hyper_opt


def _cnn_lstm_model(input_length,
                    num_classes,
                    num_features,
                    embedding_matrix,
                    embedding_dim,
                    filters_num=512,
                    filter_sizes=None,
                    dropout_rate=0.5):
    if filter_sizes is None:
        filter_sizes = [5]
    op_units, op_activation = num_classes, 'softmax'

    model = Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_length,
                        weights=[embedding_matrix],
                        trainable=False
                        ))

    model.add(Bidirectional(LSTM(units=int(embedding_dim / 2), return_sequences=True), input_shape=(-1, embedding_dim)))

    model.add(Flatten())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=op_units, activation=op_activation))

    loss = 'sparse_categorical_crossentropy'
    optimizer = Adam()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()
    return model


class BosweClassifier(Classifier):

    def __init__(self, *args, **kwargs):
        super(BosweClassifier, self).__init__()
        self._model_args = args
        self._model_kwargs = kwargs
        self._model = None

    def fit(self, dataset, *args, **kwargs):
        x, y = dataset
        self._model = _cnn_lstm_model(
            *[*self._model_args, *args],
            **{**self._model_kwargs, **kwargs}
        )
        self._model.fit(
            x, y,
            shuffle=True
        )

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data)

    @staticmethod
    def hyper_params_search(dataset, *args, **kwargs):
        x, y = dataset
        hyper_space = {

        }
        hyper_params = hyper_opt(x, y, {}, hyper_space, BosweClassifier, kappa, max_evals=100)
        return hyper_params
