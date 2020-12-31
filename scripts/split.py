"""
Split data into train (research) and test (development) sets

Strategy:
    total: 588k
    research: 10% = 58.8k
    development: 90% = 529.2k
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

p = Path(__file__).parents[1]

df = pd.read_csv(p.joinpath('data', 'raw', 'train.csv'))

res, dev = train_test_split(df, test_size=0.9, random_state=0)

res.to_pickle(p.joinpath('data', 'interim', 'research.pkl'))
dev.to_pickle(p.joinpath('data', 'interim', 'development.pkl'))