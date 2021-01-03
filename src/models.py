from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

MAX_ITER = 1e5

MODELS = [
    (
        'regression',
        lm.LinearRegression(),
        None,
    ),
    (
        'ridge',
        lm.Ridge(
            random_state=0,
            solver='saga',
            max_iter=MAX_ITER
        ),
        {'clf__alpha': [1e-3, 1e-2, 1e-1, 1e0]},
    ),
    (
        'lasso',
        lm.Lasso(
            random_state=0,
            max_iter=MAX_ITER
        ),
        {'clf__alpha': [1e-3, 1e-2, 1e-1, 1e0]},
    ),
    (
        'elastic_net',
        lm.ElasticNet(
            random_state=0,
            max_iter=MAX_ITER
        ),
        {
            'clf__alpha': [1e-3, 1e-2, 1e-1, 1e0],
            'clf__l1_ratio': [0.2, 0.5, 0.8]
        },
    ),
    (
        'decision_tree',
        DecisionTreeRegressor(random_state=0),
        None,
    ),
    (
        'light_gbm',
        LGBMRegressor(random_state=0),
        {
            'clf__boosting_type': ['gbdt', 'dart', 'goss'],
            'clf__n_estimators': [100, 200, 300]
        },
    ),
]