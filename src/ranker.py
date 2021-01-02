import pandas as pd
import numpy as np
import typing as t
from scipy import stats as ss


class Ranker:
    """ Ranks features. Uses Spearman's correlation """

    @staticmethod
    def _spearmanR(x: pd.Series, y: pd.Series, /) -> t.Tuple[float, float]:
        r, pval = ss.spearmanr(x, y)
        p = -np.log10(pval)
        return r * 100, 1000 if p == np.inf else p
    
    def rank(self, X: pd.DataFrame, y: pd.Series, /) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            data={v: self._spearmanR(X[v], y) for v in X},
            orient='index',
            columns=['Association Strength', 'Statistical Significance']
        )