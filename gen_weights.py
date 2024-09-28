import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpt import MarkowitzPortfolio

market_conditions = ['0', '1', '2', '3', '4']

# Sector ETF Name -> Ticker mapping
to_ticker = {
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Financial': 'XLF',
    'Health Care': 'XLV',
    'Industrial': 'XLI',
    'Technology': 'XLK',
    'Materials': 'XLB',
    'Utilities': 'XLU'
}

# Sector ETF Ticker -> Name mapping
to_name = {
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLF': 'Financial',
    'XLV': 'Health Care',
    'XLI': 'Industrial',
    'XLK': 'Technology',
    'XLB': 'Materials',
    'XLU': 'Utilities'
}

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
