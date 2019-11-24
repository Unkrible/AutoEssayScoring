import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, space_eval, tpe, fmin

from common import log
from constant import ESSAY_LABEL


def hyper_opt(X, y, params, space, classifier_class, metric, is_larger_better=True, **kwargs):
    """
    Use bayesian opt to search hyper parameters
    :param X: dataset
    :param y: labels
    :param params: fix params
    :param space: hyper parameters search space
    :param classifier_class: model class
    :param metric: function to calculate metric, should be func(y_true, y_pred, *args, **kwargs)
    :param is_larger_better: whether the larger the metric, the better
    :return: best hyper parameters
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    def objective(hyperparams):
        classifier = classifier_class(**{**params, **hyperparams})
        classifier.fit((X_train, y_train))
        y_pred = classifier.predict(X_val)
        score = metric(y_val[ESSAY_LABEL].tolist(), y_pred)
        if is_larger_better:
            score = -score
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    kwargs['max_evals'] = 50 if 'max_evals' not in kwargs else kwargs['max_evals']
    best = fmin(
        fn=objective,
        space=space,
        trials=trials,
        algo=tpe.suggest,
        verbose=1,
        rstate=np.random.RandomState(66),
        **kwargs
    )

    hyperparams = space_eval(space, best)

    log(f"{metric.__name__} = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

    return hyperparams
