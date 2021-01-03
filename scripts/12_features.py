"""
Get selected number of features. See `11_feature_selection.py`
"""

import joblib
import json
import pandas as pd
from pathlib import Path

p = Path(__file__).parents[1]

rnk = joblib.load(p.joinpath('src', 'meta', 'Ranking.pkl'))

json.dump(
    obj=rnk[rnk['Group'] == 1].index.to_list(),
    fp=open(file=p.joinpath('src', 'meta', 'SelectedFeatures.json'), mode='w'),
    indent=4
)