"""
Utilities
"""

import time
import typing as t
import pandas as pd
import numpy as np
from scipy import stats as ss


def skewTest(
        vals: t.Union[np.array, pd.Series],
        method: t.Callable,
        /, **kws) -> t.Tuple[float, float, float]:
    start = time.time()
    v = method(vals, **kws)
    _, p = ss.skewtest(v)
    delta = time.time() - start
    return {'Insignificance': 1/-np.log10(p[0]), 'Time': delta * 1000}