import joblib
import json
import typing as t
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pathlib import Path

p = Path(__file__).parents[1]

# Setup for loading modules in `src`
import sys
sys.path.append(str(p))
from src.logger import LOGGER
from src.models import MODELS
from src import estimators as e
from src.preprocessors import TargetPreprocessor


Estimator = t.TypeVar('Estimator')

def make_pipeline(model: Estimator) -> Pipeline:
    return Pipeline([
        ('grouper', e.CategoricalGrouper()),
        ('encoder', e.CategoricalEncoder()),
        ('clf', model)
    ])


def fit(name: str,
        model: Estimator,
        params: t.Mapping[str, float],
        filename: Path,
        X: pd.DataFrame,
        y: pd.Series, /) -> None:
    LOGGER.info(f'Run {name}')
    m = (
        make_pipeline(model) if params is None
        else GridSearchCV(
            estimator=make_pipeline(model),
            param_grid=params,
            scoring='neg_root_mean_squared_error',
            n_jobs=3,
            cv=5
        )
    ).fit(X, y)
    joblib.dump(
        value=m if params is None else m.best_estimator_,
        filename=filename
    )


LOGGER.info('Load data')
df = pd.read_pickle(p.joinpath('data', 'interim', 'development.pkl'))

LOGGER.info('Load selected features')
FEATURES = json.load(open(file=p.joinpath('src', 'meta', 'SelectedFeatures.json'), mode='r'))

LOGGER.info('Get X and y')
X = df.filter(FEATURES)
y = df['loss'].copy()

LOGGER.info('Process target')
y = TargetPreprocessor().fit_transform(y)

# Fitting
for name, model, params in MODELS:
    f = p.joinpath('models', f'{name}.pkl')
    if f.is_file():
        LOGGER.info(f'{name} on disk. Skip')
        continue
    fit(name, model, params, f, X, y)