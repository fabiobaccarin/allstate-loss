"""
Ranking of features with minimum preprocessing (raw features)
"""

import json
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib import pyplot as plt
from sklearn.preprocessing import (
    power_transform, quantile_transform, scale, StandardScaler, FunctionTransformer
)
from sklearn.pipeline import make_pipeline
from pandas_profiling import ProfileReport
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import estimators as e
from src.ranker import Ranker


A4_DIMS = (11.7, 8.27)

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

LOGGER.info('Make raw X and profile')
X = catf.join(contf)
del catf, contf
ProfileReport(
    df=X.join(y),
    minimal=True,
    title='Data with minimum preprocessing',
    progress_bar=False
).to_file(p.joinpath('reports', 'profiles', 'raw.html'))

LOGGER.info('Create ranking')
ranking = Ranker().rank(X, y)

LOGGER.info('Figure 3: feature ranking')
fig, ax = plt.subplots(figsize=A4_DIMS)
ax = sns.scatterplot(
    x=ranking['Statistical Significance'],
    y=ranking['Association Strength'],
    s=100,
)
ax.set(
    title='Figure 3: Volcano plot for features',
    xlabel='Statistical Significance (-log10(p-value))',
    ylabel='Association Strength (%)'
)
ax.axvline(x=0, color='black', lw=2)
ax.axhline(y=0, color='black', lw=2)
plt.tight_layout()
fig.savefig(
    fname=p.joinpath('reports', 'figures', '03FeatureRanking.png'),
    dpi=800,
    format='png'
)
plt.close(fig)

LOGGER.info('Table 4: feature ranking')
ranking['Rank'] = ranking['Association Strength'].abs().rank(ascending=False)
ranking['Group'] = np.ceil(ranking['Rank'].div(5))
ranking.sort_values('Rank').to_html(
    buf=p.joinpath('reports', 'tables', '04FeatureRanking.html'),
    float_format='{:.2f}'.format,
    bold_rows=False
)