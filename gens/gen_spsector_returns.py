import pandas as pd
import numpy as np
from utils import to_ticker

sector_names = list(to_ticker.keys())

stocks = pd.read_csv('./clean_data/stocks.csv')
spreturns = pd.read_csv('./clean_data/stocks_log_return.csv', index_col=0)

sector_components = {}
for sector in sector_names:
    sector_components[sector] = stocks[stocks['Sector']
                                       == sector]['Symbol'].values


spreturns.index = pd.to_datetime(spreturns.index)

# For each sector, calculate the average return of its components on each day
spreturns[sector_components['Health Care']]

sector_avg_returns = {}

for sector in sector_names:
    sector_avg_returns[sector] = spreturns[sector_components[sector]].mean(
        axis=1)

sector_avg_returns = pd.DataFrame(sector_avg_returns)

# Convert sector names to tickers for convenience
sector_avg_returns.columns = [to_ticker[sector]
                              for sector in sector_avg_returns.columns]


# Convert the index to datetime.date type
sector_avg_returns.index = sector_avg_returns.index.date

# Set an index name
sector_avg_returns.index.name = 'Date'

# Save the sector average returns to a CSV file
sector_avg_returns.to_csv('./clean_data/sp500_sector_avg_returns.csv')
