# Import the necessary modules
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import os

class StockDataProcessor:
    """
    A class to process stock data, filter based on a cutoff date, download adjusted close prices, and calculate log returns.
    """

    def __init__(self, input_file, output_folder, start_date, end_date):
        """
        Initialize the StockDataProcessor with file paths and date range for stock data processing.

        Args:
        - input_file (str): Path to the input CSV file containing stock data.
        - output_folder (str): Folder to save the processed data.
        - start_date (str): The cutoff start date for filtering the stock data.
        - end_date (str): End date for price data download from Yahoo Finance.
        """
        self.input_file = input_file
        self.output_folder = output_folder
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = end_date

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def filter_stocks_before_cutoff(self):
        """
        Filters stock data to include companies added before the cutoff date and modifies ticker symbols.
        """
        stocks = pd.read_csv(self.input_file)

        # Keep only relevant columns and rename them
        stocks = stocks[['Symbol', 'Security', 'GICS Sector', 'Date added']]
        stocks.columns = ['Symbol', 'Security', 'Sector', 'Date added']

        # Convert 'Date added' to datetime and filter
        stocks['Date added'] = pd.to_datetime(stocks['Date added'])
        stocks = stocks[stocks['Date added'] <= self.start_date].reset_index(drop=True)

        # Replace '.' in the 'Symbol' column with '-'
        stocks['Symbol'] = stocks['Symbol'].str.replace('.', '-')

        # Save filtered data
        output_file = os.path.join(self.output_folder, 'stocks.csv')
        stocks.to_csv(output_file, index=False)
        print(f"Filtered stock data saved to {output_file}")

        return stocks

    def download_price_data_and_calculate_log_returns(self, stocks):
        """
        Downloads adjusted close price data from Yahoo Finance and calculates log returns.

        Args:
        - stocks (pd.DataFrame): The filtered stock data containing ticker symbols.
        """
        assets = list(stocks['Symbol'])
        print(f"Downloading price data for {len(assets)} stocks from Yahoo Finance...")

        price_data = yf.download(assets, start=self.start_date, end=self.end_date)['Adj Close']

        # Save price data
        price_output_file = os.path.join(self.output_folder, 'price_data.csv')
        price_data.to_csv(price_output_file)
        print(f"Price data saved to {price_output_file}")

        # Calculate log returns
        print("Calculating log returns...")
        stocks_log_ret = (np.log(price_data) - np.log(price_data.shift(1))) * 100
        stocks_log_ret = stocks_log_ret.dropna()

        # Save log returns data
        log_ret_output_file = os.path.join(self.output_folder, 'stocks_log_return.csv')
        stocks_log_ret.to_csv(log_ret_output_file)
        print(f"Log returns data saved to {log_ret_output_file}")


if __name__ == "__main__":
    # Parameters for stock data processing
    input_file = './raw_data/S&P.csv'
    output_folder = './clean_data'
    start_date = '2011-12-30'  # Adjust this for replicability
    end_date = '2024-07-31'

    # Create an instance of StockDataProcessor
    processor = StockDataProcessor(input_file, output_folder, start_date, end_date)

    # Step-by-step: filter stocks and download data
    stocks = processor.filter_stocks_before_cutoff()  # Filter stocks based on the cutoff date
    processor.download_price_data_and_calculate_log_returns(stocks)  # Download price data and calculate log returns
