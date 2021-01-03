import json
import typing as t
import pandas as pd
from src import estimators as e
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path


p = Path(__file__).parents[1]

NO_VAR_FEATURES = json.load(open(file=p.joinpath('src', 'meta', 'NoVariance.json'), mode='r'))

CORRELATED = json.load(open(file=p.joinpath('src', 'meta', 'Correlated.json'), mode='r'))
CORRELATED_CAT = [c for c in CORRELATED if c.startswith('cat')]
CORRELATED_CONT = [c for c in CORRELATED if c.startswith('cont')]
del CORRELATED


class TargetPreprocessor(BaseEstimator, TransformerMixin):
    """ Stabilizes the variance of the target """

    def __init__(self):
        self.preprocessor = PowerTransformer()

    def fit(self, X: pd.Series, y: t.Optional[pd.Series] = None) -> 'TargetPreprocessor':
        self.preprocessor.fit(X.values.reshape(-1, 1), y)
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return pd.Series(
            data=self.preprocessor.transform(X.values.reshape(-1, 1)).flatten(),
            name='loss',
            index=X.index
        )

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        return pd.Series(
            data=self.preprocessor.inverse_transform(X.values.reshape(-1, 1)).flatten(),
            name='loss',
            index=X.index
        )


class Preprocessor(BaseEstimator, TransformerMixin):
    """ Applies preprocessing of features, with a pipeline for different types of features """
    
    def __init__(self):
        self.catPreprocessor = make_pipeline(
            e.CategoricalGrouper(),
            e.CategoricalEncoder()
        )
        self.catPCA = make_pipeline(
            e.CategoricalGrouper(),
            e.CategoricalEncoder(),
            PCA(n_components=6, random_state=0)
        )
        self.contPreprocessor = make_pipeline(
            QuantileTransformer(output_distribution='normal', random_state=0),
            StandardScaler()
        )
        self.contPCA = make_pipeline(
            QuantileTransformer(output_distribution='normal', random_state=0),
            StandardScaler(),
            PCA(n_components=6, random_state=0)
        )

    @staticmethod
    def getCatColumns(X: pd.DataFrame) -> t.List[str]:
        return (
            X.filter(like='cat')
            .drop(labels=NO_VAR_FEATURES+CORRELATED_CAT, axis=1).columns
            .to_list()
        )

    @staticmethod
    def getCatCorrelatedCols(X: pd.DataFrame) -> t.List[str]:
        return X.filter(CORRELATED_CAT).columns.to_list()

    @staticmethod
    def getContColumns(X: pd.DataFrame) -> t.List[str]:
        return X.filter(like='cont').drop(labels=CORRELATED_CONT, axis=1).columns.to_list()

    @staticmethod
    def getContCorrelatedCols(X: pd.DataFrame) -> t.List[str]:
        return X.filter(CORRELATED_CONT).columns.to_list()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, /) -> 'Preprocessor':
        self.catCols = self.getCatColumns(X)
        self.catPreprocessor.fit(X.filter(self.catCols), y)
        
        self.catPCACols = self.getCatCorrelatedCols(X)
        self.catPCA.fit(X.filter(self.catPCACols), y)
        
        self.contCols = self.getContColumns(X)
        self.contPreprocessor.fit(X.filter(self.contCols))

        self.contPCACols = self.getContCorrelatedCols(X)
        self.contPCA.fit(X.filter(self.contPCACols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        catf = pd.DataFrame(
            data=self.catPreprocessor.transform(X.filter(self.catCols)),
            columns=self.catCols,
            index=X.index
        )
        catPCAf = pd.DataFrame(
            data=self.catPCA.transform(X.filter(self.catPCACols)),
            columns=[f'cat_pca{i:02d}' for i in range(1, 7)],
            index=X.index
        )
        contf = pd.DataFrame(
            data=self.contPreprocessor.transform(X.filter(self.contCols)),
            columns=self.contCols,
            index=X.index
        )
        contPCAf = pd.DataFrame(
            data=self.contPCA.transform(X.filter(self.contPCACols)),
            columns=[f'cont_pca{i:02d}' for i in range(1, 7)],
            index=X.index
        )
        return catf.join(catPCAf).join(contf).join(contPCAf)