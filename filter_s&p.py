# Import the necessary modules
import pandas as pd
import datetime
import os


def filter_stocks_before_2012(input_file, output_file, start_date):
    """
    Filters the stock data to only include entries added before 2012 and formats ticker symbols.

    Args:
    - input_file (str): Path to the input CSV file containing stock data.
    - output_file (str): Path to save the filtered and formatted stock data.
    - start_date (datetime.datetime): The cutoff date for filtering the stock data.
    """

    # Load stock data
    stocks = pd.read_csv(input_file)

    # Keep only relevant columns and rename them
    stocks = stocks[['Symbol', 'Security', 'GICS Sector', 'Date added']]
    stocks.columns = ['Symbol', 'Security', 'Sector', 'Date added']

    # Convert 'Date added' column to datetime format
    stocks['Date added'] = pd.to_datetime(stocks['Date added'])

    # Filter for stocks added before the cutoff date
    stocks = stocks[stocks['Date added'] <= start_date].reset_index(drop=True)

    # Replace '.' in the 'Symbol' column with '-'
    stocks['Symbol'] = stocks['Symbol'].str.replace('.', '-')

    # Ensure 'clean_data' directory exists, if not, create it
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the filtered data to a new CSV file
    stocks.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


if __name__ == "__main__":
    # Define input file path and output file path in the 'clean_data' folder
    input_file = './raw_data/S&P.csv'
    output_file = './clean_data/stocks.csv'

    # Call the function to filter stocks and save the results
    filter_stocks_before_2012(input_file, output_file,
                              start_date=datetime.datetime(2012, 1, 1))
