"""
Pandas profile of datasets
"""

import pandas as pd
from pandas_profiling import ProfileReport
from pathlib import Path

p = Path(__file__).parents[1]

# To load project modules
import sys; sys.path.append(str(p))

from src.logger import LOGGER

LOGGER.info('Load data')
res = pd.read_pickle(p.joinpath('data', 'interim', 'research.pkl'))
dev = pd.read_pickle(p.joinpath('data', 'interim', 'development.pkl'))


LOGGER.info('Profile on research')
(
    ProfileReport(
        df=res,
        config_file=p.joinpath('src', 'ProfileConf.yml'),
        title='Research dataset',
        dataset={'description': r'Research dataset (10% of total data)'}
    )
    .to_file(p.joinpath('reports', 'profiles', 'research.html'))
)

LOGGER.info('Profile on development')
(
    ProfileReport(
        df=dev,
        config_file=p.joinpath('src', 'ProfileConf.yml'),
        title='Development dataset',
        dataset={'description': r'Development dataset (10% of total data)'}
    )
    .to_file(p.joinpath('reports', 'profiles', 'development.html'))
)