import joblib
import itertools
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, RepeatedKFold
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import preprocessors as pp
from src.ranker import Ranker


CROSS_VAL_OPTS = {
    'scoring': 'neg_root_mean_squared_error',
    'cv': RepeatedKFold(n_splits=5, n_repeats=20),
    'n_jobs': 3,
    'return_train_score': False,
    'return_estimator': False,
}

r = Ridge(random_state=0, solver='saga', max_iter=1e5)

LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
X = df.drop(labels='loss', axis=1)
y = df['loss'].copy()

LOGGER.info('Process target')
tp = pp.TargetPreprocessor() # will be needed to convert back to y original units
y = tp.fit_transform(y)

LOGGER.info('Process X')
X = pp.Preprocessor().fit_transform(X, y)

LOGGER.info('Prepare dummy regressor')
dummyRMSE = mean_squared_error(
    y_true=y,
    y_pred=DummyRegressor(strategy='mean').fit(X, y).predict(X),
    squared=False
)

LOGGER.info('Load ranking')
rnk = joblib.load(p.joinpath('src', 'meta', 'Ranking.pkl'))

LOGGER.info('Get groups of features')
groups = [
    rnk[rnk['Group'] == i].index.to_list()
    for i in range(1, int(rnk['Group'].max() + 1))
]

bm = pd.DataFrame()
for g in range(len(groups)):
    features = list(itertools.chain.from_iterable(groups[:g+1]))
    LOGGER.info(f"Selection - group {' + '.join([str(i+1) for i in range(g+1)])} = {len(features)} features")
    label = f'selection{g+1:02d} ({len(features):02d} features)'
    cv = dict(cross_validate(estimator=r, X=X.filter(features), y=y, **CROSS_VAL_OPTS))
    bm = pd.concat([bm, pd.DataFrame(cv).assign(experiment=label)])
tbl = (
    bm.assign(
        fit_time=lambda df: df['fit_time'].mul(1000),
        score_time=lambda df: df['score_time'].mul(1000),
        test_score=lambda df: tp.inverse_transform(df['test_score'].add(dummyRMSE))
    )
    .groupby('experiment')
)

LOGGER.info('Table 6: Feature selection')
(
    tbl.mean().add_suffix('_mean')
    .join(tbl.std().div(np.sqrt(CROSS_VAL_OPTS['cv'].get_n_splits())).add_suffix('_stderr'))
    .assign(cer=lambda df: (
        df.filter(['fit_time_mean', 'score_time_mean']).sum(axis=1).div(df['test_score_mean'])
    ))
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
        buf=p.joinpath('reports', 'tables', '06FeatureGroups.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    )
)