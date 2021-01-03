import pandas as pd
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))
from src.logger import LOGGER
from src import preprocessors as pp
from pandas_profiling import ProfileReport

LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
X = df.drop(labels='loss', axis=1)
y = df['loss'].copy()

LOGGER.info('Process target')
y = pp.TargetPreprocessor().fit_transform(y)

LOGGER.info('Process X')
X = pp.Preprocessor().fit_transform(X, y)

LOGGER.info('Profile final data')
ProfileReport(
    df=X.join(y),
    config_file=p.joinpath('src', 'ProfileConf.yml'),
    title='Final dataset'
).to_file(p.joinpath('reports', 'profiles', 'final.html'))