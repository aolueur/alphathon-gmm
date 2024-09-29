import pandas as pd
from utils import DataMerger

if __name__ == "__main__":
    # File paths for input and output
    labels_file = './clean_data/labels.csv'
    regimes_file = './clean_data/3_market_regimes.txt'
    output_file_partial = './clean_data/combined_partial.csv'
    output_file_full = './clean_data/combined_full.csv'

    # Create an instance of DataMerger and process the data
    merger = DataMerger(labels_file, regimes_file)
    merger.load_data()  # Load the data from the CSV files
    # Save the merged results
    merger.save_merged_data(output_file_partial, output_file_full)
