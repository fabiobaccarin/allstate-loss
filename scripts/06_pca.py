"""
PCA analysis of continuous features
"""

import pandas as pd
import seaborn as sns; sns.set(style='whitegrid')
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER


A4_DIMS = (11.7, 8.27)

LOGGER.info('Load data')
X = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl')).filter(like='cont')

LOGGER.info(r'Figure 1: Correlations above 75% for continuous features')
corr = X.corr()
col_order = corr.columns[AgglomerativeClustering().fit_predict(corr).argsort()]
fig, _ = plt.subplots(figsize=A4_DIMS)
ax = sns.heatmap(data=corr.loc[col_order, col_order].gt(0.75), cmap='Greys', cbar=False)
ax.set(title=r'Figure 1: Correlations above 75% for continuous features')
plt.tight_layout()
fig.savefig(
    fname=p.joinpath('reports', 'figures', '01ContFeaturesCorr75Matrix.png'),
    dpi=800,
    format='png'
)
plt.close(fig=fig)

LOGGER.info('Prepare features for PCA')
X = pd.DataFrame(
    data=make_pipeline(
        QuantileTransformer(output_distribution='normal', random_state=0),
        StandardScaler()
    ).fit_transform(X),
    columns=X.columns,
    index=X.index
)

LOGGER.info('Table 3: Determination of number of PCA components')
pca = PCA(random_state=0).fit(X)
(
    pd.Series({
        f'Component {c}': exp_var
        for c, exp_var in enumerate(pca.explained_variance_ratio_, start=1)
    }).to_frame('ExplainedVariance')
    .assign(AccumulatedExplainedVariance=lambda df: df['ExplainedVariance'].cumsum())
    .to_html(
        buf=p.joinpath('reports', 'tables', '03NumComponentsPCA.html'),
        float_format='{:.2%}'.format,
        bold_rows=False
    )
)