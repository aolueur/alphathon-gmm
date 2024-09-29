"""
This module generates the data to be fed into the GMM model.

The processed data is saved in the './clean_data/factor_returns.csv' file.
"""
import sys
from utils import generate_data


if __name__ == '__main__':
    filepath_return = [
        './raw_data/F-F_Momentum_Factor_daily.csv',
        './raw_data/F-F_Research_Data_5_factors_2x3_daily.csv'
    ]
    filepath_price = ['./raw_data/CPIAUCSL.csv']

    generate_data(filepath_return, filepath_price)
