import pandas as pd
import yfinance as yf
import numpy as np


class TimeSeriesDataHandler:
    def __init__(self, filepath_return, filepath_price):
        """Initialize the TimeSeriesDataHandler

        Args:
            filepaths: list of str.
                The list of filepaths to the datasets.
        """
        self.filepath_return = filepath_return
        self.filepath_price = filepath_price
        self.datasets = []

    def load_return_data(self):
        for filepath in self.filepath_return:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, format='%Y%m%d')
            data.index = data.index.strftime("%Y-%m-%d")
            self.datasets.append(data)

    def load_price_data(self):
        for filepath in self.filepath_price:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, format='%Y%m%d')
            data.index = data.index.strftime("%Y-%m-%d")
            data = np.log(data) - np.log(data.shift(1))
            self.datasets.append(data)

    def clean_data(self):
        for data in self.datasets:
            data.replace([-99.99, -999], pd.NA, inplace=True)
            data.fillna(method='ffill', inplace=True)

    def filter_before_date(self, date: str):
        """Filter data before a given date, yyyymmdd"""
        for i in range(len(self.datasets)):
            self.datasets[i] = self.datasets[i][self.datasets[i].index >= date]

    def download_ticker_data(self, ticker: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download ticker data using yfinance

        Args:
            ticker: list of str.
                List of the ticker symbols. For example, ['AAPL', 'MSFT'].
            start_date: str.
                The start date in the format 'YYYY-MM-DD'.
            end_date: str.
                The end date in the format 'YYYY-MM-DD'."""

        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        ticker_data.index = ticker_data.index.strftime("%Y-%m-%d")
        close = ticker_data['Adj Close']
        return np.log(close)-np.log(close.shift(1))

    def add_ticker_data(self, tickers: list[str], start_date: str, end_date: str):
        """Add ticker data to the datasets

        Args:
            ticker: str.
                List of the ticker symbols. For example, ['AAPL', 'MSFT'].
            start_date: str.
                The start date in the format 'YYYY-MM-DD'.
            end_date: str.
                The end date in the format 'YYYY-MM-DD'."""
        ticker_data = self.download_ticker_data(tickers, start_date, end_date)
        for ticker in tickers:
            if ticker in ticker_data.columns:
                self.datasets.append(ticker_data[ticker].rename(ticker))

    def merged_datasets(self) -> pd.DataFrame:
        
        merged_data = self.datasets[0]
        for data in self.datasets[1:]:
            merged_data = pd.merge(
                merged_data, data, left_index=True, right_index=True, how='left')
        merged_data.fillna(method='ffill', inplace=True)
        return merged_data[19:-1]
