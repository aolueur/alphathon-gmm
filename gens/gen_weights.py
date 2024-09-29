import sys
from utils import to_ticker, to_name
from mpt import MarkowitzPortfolio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


market_conditions = ['0', '1', '2', '3', '4']


# Load data
sector_etf_names = list(to_name.values())
mean_returns = pd.read_csv('clean_data/mean.csv',
                           index_col=0).loc[sector_etf_names]

# Read convariance matrices
cov_matrices = {}

for condition in market_conditions:
    cov_matrices[condition] = pd.read_csv(
        f'./clean_data/covariance_{condition}.csv', index_col=0)
    # filter out rows and columns that represent sector ETFs
    cov_matrices[condition] = cov_matrices[condition].loc[sector_etf_names,
                                                          sector_etf_names]

print(cov_matrices['0'])
weights = {}  # Store optimal weights for each market condition

for condition in market_conditions:
    # Create MarkowitzPortfolio object
    mp = MarkowitzPortfolio(
        mean_returns[condition], cov_matrices[condition], 0)

    # Calculate optimal weights
    weights[condition] = mp.optimize()
    print(f'Optimal weights for condition {condition}:')

pd.DataFrame(weights, index=sector_etf_names).to_json(
    './clean_data/optimal_weights.json')
