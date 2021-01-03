import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path


p = Path(__file__).parents[1]

# Setup to load modules in `src`
import sys
sys.path.append(str(p))
from src.logger import LOGGER
from src.preprocessors import TargetPreprocessor


CROSS_VAL_OPTS = {
    'scoring': 'neg_root_mean_squared_error',
    'cv': RepeatedKFold(n_splits=5, n_repeats=20),
    'n_jobs': 3,
    'return_train_score': False,
    'return_estimator': False,
}

LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'development.pkl'))

LOGGER.info('Load selected features')
FEATURES = json.load(open(file=p.joinpath('src', 'meta', 'SelectedFeatures.json'), mode='r'))

LOGGER.info('Get X and y')
X = df.filter(FEATURES)
y = df['loss'].copy()

LOGGER.info('Process target')
tp = TargetPreprocessor() # will be needed to convert back to y original units
y = tp.fit_transform(y)

LOGGER.info('Load models')
MODELS = {model.stem: joblib.load(model) for model in p.joinpath('models').iterdir()} 

LOGGER.info('Prepare dummy regressor')
dummyRMSE = mean_squared_error(
    y_true=y,
    y_pred=DummyRegressor(strategy='mean').fit(X, y).predict(X),
    squared=False
)

# Evaluate
bm = pd.DataFrame()
for name, model in MODELS.items():
    LOGGER.info(f'Run {name}')
    cv = dict(cross_validate(estimator=model, X=X, y=y, **CROSS_VAL_OPTS))
    bm = pd.concat([bm, pd.DataFrame(cv).assign(model=name)])
tbl = (
    bm.assign(
        fit_time=lambda df: df['fit_time'].mul(1000),
        score_time=lambda df: df['score_time'].mul(1000),
        test_score=lambda df: tp.inverse_transform(df['test_score'].add(dummyRMSE))
    )
    .groupby('model')
)

LOGGER.info('Table 7: Model performance comparison')
(
    tbl.mean().add_suffix('_mean')
    .join(tbl.std().div(np.sqrt(CROSS_VAL_OPTS['cv'].get_n_splits())).add_suffix('_stderr'))
    .assign(cer=lambda df: (
        df.filter(['fit_time_mean', 'score_time_mean']).sum(axis=1).div(df['test_score_mean'])
    ))
    .sort_values('cer')
    .rename(columns={
        'fit_time_mean': 'Mean fit time (ms)',
        'score_time_mean': 'Mean score time (ms)',
        'test_score_mean': 'Mean reduction in test RMSE ($)',
        'fit_time_stderr': 'Fit time standard error (ms)',
        'score_time_stderr': 'Score time standard error (ms)',
        'test_score_stderr': 'Reduction in test RMSE standard error ($)',
        'cer': 'Cost-effectiveness ratio'
    })
    .to_html(
        buf=p.joinpath('reports', 'tables', '07ModelPerformanceComparison.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    )
)