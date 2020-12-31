"""
Cost-effectiveness analysis of transforming y for better prediction

Options:
    A. None
    B. Log
    C. Yeo-Johnson
    D. QuantileTransformer
"""

import time
import typing as t
import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform, quantile_transform
from scipy import stats as ss
from pathlib import Path


def skewTest(
        vals: t.Union[np.array, pd.Series],
        method: t.Callable, /, **kws) -> t.Tuple[float, float, float]:
    start = time.time()
    y = method(vals, **kws)
    _, p = ss.skewtest(y)
    delta = time.time() - start
    return {'Insignificance': 1/-np.log10(p[0]), 'Time': delta * 1000}


p = Path(__file__).parents[1]

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
    .to_html(
        buf=p.joinpath('reports', 'tables', '01YTransformations.html'),
        float_format='{:.2f}'.format,
        bold_rows=False
    )
)