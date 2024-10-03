# Import the necessary modules
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import os
from utils import StockDataProcessor


if __name__ == "__main__":
    # Parameters for stock data processing
    input_file = './raw_data/S&P.csv'
    output_folder = './clean_data'
    start_date = '2011-12-30'  # Adjust this for replicability
    end_date = '2024-07-31'

    # Create an instance of StockDataProcessor
    processor = StockDataProcessor(
        input_file, output_folder, start_date, end_date)

    # Step-by-step: filter stocks and download data
    # Filter stocks based on the cutoff date
    stocks = processor.filter_stocks_before_cutoff()
    processor.download_price_data_and_calculate_log_returns(
        stocks)  # Download price data and calculate log returns
