# only a test, dont run, feel free to delete

import pandas as pd
import yfinance as yf
# df = pd.read_csv('F-F_Momentum_Factor_daily.csv')
# df['Unnamed: 0'] = df['Unnamed: 0'].apply(
#     lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

# print(df)
from DataPreprocessing import TimeSeriesDataHandler

# ticker_data = yf.download(['AAPL','MSFT'], start='2015-01-01', end='2015-12-31')
# ticker_data.index = ticker_data.index.strftime("%Y-%m-%d")
# print(ticker_data.head)

    # define the file paths for the input and output files
filepaths = ['F-F_Momentum_Factor_daily.csv']
    # Create an instance of the TimeSeriesDataHandler
ts_data_handler = TimeSeriesDataHandler(filepaths)
# Load the data from the input files
ts_data_handler.load_data()
a =ts_data_handler.datasets[0]
b = yf.download('AAPL', start='2015-01-01', end='2015-12-31')['Adj Close']
b.index = b.index.strftime("%Y-%m-%d")
c = pd.merge(a,b, how= 'right', left_index = True, right_index=True)
print(c)