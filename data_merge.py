import pandas as pd

class DataMerger:
    """
    A class for merging datasets based on a common 'Date' column.
    """
    
    def __init__(self, labels_file, regimes_file):
        """
        Initialize the DataMerger with file paths for the labels and regimes data.

        Args:
        - labels_file (str): Path to the labels CSV file.
        - regimes_file (str): Path to the 5 market regimes CSV file (or .txt).
        """
        self.labels_file = labels_file
        self.regimes_file = regimes_file
        self.labels = None
        self.data = None

    def load_data(self):
        """
        Load the labels and market regimes data.
        """
        # Load the regimes data with custom column names
        self.data = pd.read_csv(self.regimes_file, sep=",", header=None)
        self.data.columns = ['Date', 'LLM_Labels']
        
        # Load the labels data
        self.labels = pd.read_csv(self.labels_file)
        
    def save_merged_data(self, output_file_partial, output_file_full):
        """
        Save the merged datasets to CSV files. Performs inner and left joins.

        Args:
        - output_file_partial (str): Path to save the partial (inner join) merged CSV.
        - output_file_full (str): Path to save the full (left join) merged CSV.
        """
        if self.labels is None or self.data is None:
            raise ValueError("Data not loaded. Call load_data() before saving merged data.")
        
        # Perform inner join and left join
        combined_partial = pd.merge(self.labels, self.data, on='Date', how='inner')
        combined_full = pd.merge(self.labels, self.data, on='Date', how='left')
        combined_full = combined_full[combined_full['Date'] >= combined_partial['Date'][0]]
        combined_full.ffill(inplace = True)
        
        # Save the merged data to CSV files
        combined_partial.to_csv(output_file_partial, index=False)
        combined_full.to_csv(output_file_full, index=False)

        print(f"Partial merged dataset saved to {output_file_partial}")
        print(f"Full merged dataset saved to {output_file_full}")

if __name__ == "__main__":
    # File paths for input and output
    labels_file = './clean_data/labels.csv'
    regimes_file = './clean_data/3_market_regimes.txt'
    output_file_partial = './clean_data/combined_partial.csv'
    output_file_full = './clean_data/combined_full.csv'

    # Create an instance of DataMerger and process the data
    merger = DataMerger(labels_file, regimes_file)
    merger.load_data()  # Load the data from the CSV files
    merger.save_merged_data(output_file_partial, output_file_full)  # Save the merged results
