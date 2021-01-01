"""
Identifies which categorical features need to be dropped due to low variance
"""

import json
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import estimators as e


# Load data
LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
X = df.filter(like='cat')
y = df['loss'].copy()

# Process data
LOGGER.info('Process data')
X = pd.DataFrame(
    data=make_pipeline(e.CategoricalGrouper(), e.CategoricalEncoder()).fit_transform(X, y),
    columns=X.columns,
    index=X.index
)

# Variance threshold analysis
LOGGER.info('Variance threshold analysis')
vt = VarianceThreshold().fit(X)
drop = [col for col in X if col not in X.columns[vt.get_support()]]

# Saving results
LOGGER.info('Saving results')
json.dump(
    obj=drop,
    fp=open(file=p.joinpath('src', 'meta', 'NoVariance.json'), mode='w'),
    indent=4
)