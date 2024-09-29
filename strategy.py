import pandas as pd
import numpy as np
from utils import to_ticker
import datetime


def calculate_annualized_returns(start_date, end_date):
    # Load the data
    with open('./clean_data/optimal_weights.json') as f:
        optimal_weights = pd.read_json(f)

    with open('./clean_data/priors.json') as f:
        priors = pd.read_json(f)

    with open('./clean_data/factor_returns.csv') as f:
        returns = pd.read_csv(f, index_col=0)

    # convert index to datetime
    priors.index = pd.to_datetime(priors.index)
    returns.index = pd.to_datetime(returns.index)

    # Filter the data to the desired date range
    returns = returns.loc[start_date:end_date]
    priors = priors.loc[start_date -
                        pd.DateOffset(days=1):end_date - pd.DateOffset(days=1)]

    # extract columns that represent sector ETFs
    sector_tickers = list(to_ticker.values())
    returns = returns[sector_tickers]

    # Get the list of dates
    dates = list(returns.index)
    print(f'Date range from {dates[0]} to {dates[-1]}')

    # Get the optimal weights for each day
    # portfolio weights from start_date to end_date
    weighted_optimal_weights = np.dot(optimal_weights, priors.T).T

    # Calculate portfolio returns for each day
    portfolio_returns = np.sum(weighted_optimal_weights * returns, axis=1)

    # Annualize the returns
    annualized_returns = np.mean(portfolio_returns) * 252

    return annualized_returns


# Example usage
START_DATE = datetime.date(2002, 1, 3)
END_DATE = datetime.date(2024, 7, 30)
annualized_returns = calculate_annualized_returns(START_DATE, END_DATE)
print(annualized_returns)