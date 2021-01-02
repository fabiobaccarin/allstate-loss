"""
Finds pathologically correlated features
"""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    power_transform, quantile_transform, scale, StandardScaler, FunctionTransformer
)
from sklearn.pipeline import make_pipeline
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import estimators as e


LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
X = df.drop(labels='loss', axis=1)
y = df['loss'].copy()

LOGGER.info('Process target')
y = pd.Series(data=power_transform(y.values.reshape(-1, 1)).flatten(), name='loss', index=y.index)

LOGGER.info('Load categorical features to drop')
noVarFeatures = json.load(open(file=p.joinpath('src', 'meta', 'NoVariance.json'), mode='r'))

LOGGER.info('Process categorical features')
catf = pd.DataFrame(
    data=make_pipeline(
        e.CategoricalGrouper(),
        e.CategoricalEncoder()
    ).fit_transform(X.filter(like='cat').drop(labels=noVarFeatures, axis=1), y),
    columns=X.filter(like='cat').drop(labels=noVarFeatures, axis=1).columns,
    index=X.index
)

LOGGER.info('Process continuous features')
contf = pd.DataFrame(
    data=scale(quantile_transform(
        X=X.filter(like='cont'),
        output_distribution='normal',
        random_state=0
    )),
    columns=X.filter(like='cont').columns,
    index=X.index
)

LOGGER.info('Find correlations')
corr = catf.join(contf).corr()
del catf, contf
np.fill_diagonal(corr.values, 0)

LOGGER.info('Persist as JSON')
json.dump(
    obj=corr.columns[corr.gt(0.75).any(axis=1)].to_list(),
    fp=open(file=p.joinpath('src', 'meta', 'Correlated.json'), mode='w'),
    indent=4
)