"""
Cost-effectiveness analysis of transforming y for better prediction

Options:
    A. Log
    B. Yeo-Johnson
    C. QuantileTransformer
"""

import time
import typing as t
import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform, quantile_transform
from scipy import stats as ss
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.utils import skewTest

loss = pd.read_csv(p.joinpath('data', 'raw', 'train.csv'), usecols=['loss'])

(
    pd.DataFrame({
        'Log': skewTest(loss, np.log),
        'Yeo-Johnson': skewTest(loss, power_transform),
        'Quantile Transformer': skewTest(
            loss,
            quantile_transform,
            output_distribution='normal',
            random_state=0
        ),
    }).T
    .assign(CostEffectivenessRatio=lambda df: df['Time'].div(df['Insignificance']))
    .apply(lambda s: s.explode())
    .sort_values('CostEffectivenessRatio')
    .to_html(
        buf=p.joinpath('reports', 'tables', '01YTransformations.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    )
)