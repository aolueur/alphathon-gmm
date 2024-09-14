from DataPreprocessing import TimeSeriesDataHandler


def main():
    # define the file paths for the input and output files
    filepath_return = ['F-F_Momentum_Factor_daily.csv',
                 'F-F_Research_Data_5_factors_2x3_daily.csv']
    filepath_price = ['CPIAUCSL.csv']
    # Create an instance of the TimeSeriesDataHandler
    ts_data_handler = TimeSeriesDataHandler(filepath_return,filepath_price)

    # Load the return data from the input files
    ts_data_handler.load_return_data()

    # Load the data price from the input files
    ts_data_handler.load_price_data()

    # Preprocess the data
    ts_data_handler.clean_data()

    # Filter data to remove entries before a certain year
    ts_data_handler.filter_before_date('19981222')

    # Download ticker data and add it to the datasets
    # ts_data_handler.download_ticker_data(['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV',
    #                                       'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', '^IQHGCPI'], '1984-01-01', '2023-07-31')

    # Ddd ticker
    ts_data_handler.add_ticker_data(['XLY', 'XLP', 'XLE', 'XLF', 'XLV',
                                    'XLI', 'XLB', 'XLK', 'XLU'], '1984-01-01', '2024-07-31')
    # Get the merged datasets
    
    df = ts_data_handler.merged_datasets()

    with open('cleaned_data.csv', 'w') as csv_file:
        df.to_csv(path_or_buf=csv_file)

    print(df)

if __name__ == '__main__':
    main()
