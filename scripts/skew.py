"""
Analyses skewness for continuous features

Options:
    A. Log
    B. Yeo-Johnson
    C. QuantileTransformer
"""

import json
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import power_transform, quantile_transform
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src.utils import skewTest


LOGGER.info('Load data')
X = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl')).filter(like='cont')

LOGGER.info('Process data - Logarithm')
A = (
    pd.DataFrame(X.apply(skewTest, args=(np.log1p,)).to_list())
    .assign(Transformation='Logarithm')
    .set_index('Transformation')
)

LOGGER.info('Process data - Yeo-Johnson')
B = (
    pd.DataFrame(
        X.apply(lambda s: skewTest(np.reshape(s.values, (-1, 1)), power_transform))
        .to_list()
    )
    .apply(lambda s: s.explode().astype(float))
    .assign(Transformation='Yeo-Johnson')
    .set_index('Transformation')
)

LOGGER.info('Process data - Quantile Transform')
C = (
    pd.DataFrame(
        X.apply(lambda s: skewTest(
            np.reshape(s.values, (-1, 1)),
            quantile_transform,
            output_distribution='normal',
            random_state=0
        ))
        .to_list()
    )
    .apply(lambda s: s.explode().astype(float))
    .assign(Transformation='Quantile Transform')
    .set_index('Transformation')
)

LOGGER.info('Computing result')
(
    pd.concat([A, B, C]).reset_index().groupby('Transformation').mean()
    .assign(CostEffectivenessRatio=lambda df: df['Time'].div(df['Insignificance']))
    .sort_values('CostEffectivenessRatio')
    .to_html(
        buf=p.joinpath('reports', 'tables', '02ContTransformations.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    )
)