"""
PCA analysis of continuous features
"""

import json
import typing as t
import pandas as pd
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import (
    power_transform, quantile_transform, scale, StandardScaler, FunctionTransformer
)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import estimators as e

A4_DIMS = (11.7, 8.27)



FittedPCA = t.TypeVar('FittedPCA')


def pcaAnalysis(X: pd.DataFrame, outputName: str, /) -> FittedPCA:
    pca = PCA(random_state=0).fit(X)
    (
        pd.Series({
            f'Component {c}': exp_var
            for c, exp_var in enumerate(pca.explained_variance_ratio_, start=1)
        }).to_frame('ExplainedVariance')
        .assign(AccumulatedExplainedVariance=lambda df: df['ExplainedVariance'].cumsum())
        .to_html(
            buf=p.joinpath('reports', 'tables', outputName),
            float_format='{:.2%}'.format,
            bold_rows=False
        )
    )
    return pca


def plotPCAWeights(pca: PCA, cols: pd.Index, title: str, out: str, /) -> None:
    fig, _ = plt.subplots(figsize=A4_DIMS)
    ax = sns.heatmap(
        data=pd.DataFrame(
            data=pca.components_,
            columns=cols,
            index=[f'Component {c+1}' for c in range(len(cols))]
        ),
        cmap='RdBu',
        annot=True,
        fmt='.2f',
        cbar=False
    )
    ax.set(title=title)
    plt.tight_layout()
    fig.savefig(
        fname=p.joinpath('reports', 'figures', out),
        dpi=800,
        format='png'
    )
    plt.close(fig=fig)


LOGGER.info('Load correlated features')
CORRELATED = json.load(open(file=p.joinpath('src', 'meta', 'Correlated.json'), mode='r'))

LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
X = df.filter(CORRELATED)
y = df['loss'].copy()

LOGGER.info('Process target')
y = pd.Series(data=power_transform(y.values.reshape(-1, 1)).flatten(), name='loss', index=y.index)

LOGGER.info('Process categorical features')
catf = pd.DataFrame(
    data=make_pipeline(
        e.CategoricalGrouper(),
        e.CategoricalEncoder()
    ).fit_transform(X.filter(like='cat'), y),
    columns=X.filter(like='cat').columns,
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

LOGGER.info(r'Figure 1: Correlations above 75%')
X = catf.join(contf)
del catf, contf
corr = X.corr()
col_order = corr.columns[AgglomerativeClustering().fit_predict(corr).argsort()]
fig, _ = plt.subplots(figsize=A4_DIMS)
ax = sns.heatmap(data=corr.loc[col_order, col_order].gt(0.75), cmap='Greys', cbar=False)
ax.set(title=r'Figure 1: Correlations above 75%')
plt.tight_layout()
fig.savefig(
    fname=p.joinpath('reports', 'figures', '01FeaturesCorr75Matrix.png'),
    dpi=800,
    format='png'
)
plt.close(fig=fig)

LOGGER.info('Table 3: Determination of number of PCA components - categorical features')
catPCA = pcaAnalysis(X.filter(like='cat'), '03NumComponentsCatPCA.html')

LOGGER.info('Figure 2: PCA weights - categorical features')
plotPCAWeights(
    catPCA,
    X.filter(like='cat').columns,
    'Figure 2: PCA weights - categorical features',
    '02CatPCAWeights.png'
)

LOGGER.info('Table 4: Determination of number of PCA components - continuous features')
contPCA = pcaAnalysis(X.filter(like='cont'), '04NumComponentsContPCA.html')

LOGGER.info('Figure 3: PCA weights - continuous features')
plotPCAWeights(
    contPCA,
    X.filter(like='cont').columns,
    'Figure 3: PCA weights - continuous features',
    '03ContPCAWeights.png'
)