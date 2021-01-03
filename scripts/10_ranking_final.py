import pandas as pd
import numpy as np
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import preprocessors as pp
from src.ranker import Ranker

LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
X = df.drop(labels='loss', axis=1)
y = df['loss'].copy()

LOGGER.info('Process target')
y = pp.TargetPreprocessor().fit_transform(y)

LOGGER.info('Process X')
X = pp.Preprocessor().fit_transform(X, y)

LOGGER.info('Create ranking')
rnk = Ranker().rank(X, y)
rnk['Rank'] = rnk['Association Strength'].abs().rank(ascending=False)
rnk['Group'] = np.ceil(rnk['Rank'].div(5))
rnk.to_pickle(p.joinpath('src', 'meta', 'Ranking.pkl'))