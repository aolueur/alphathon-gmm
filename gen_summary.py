import pandas as pd

class DataSummaryGenerator:
    """
    A class to generate summary statistics (mean and standard deviation) for factor data with labels.
    
    Attributes:
    -----------
    factor_file : str
        Path to the CSV file containing the factor data.
    label_file : str
        Path to the CSV file containing the labels.
    col_names : list of str
        List of column names for the combined factor and label data.
    output_dir : str
        Directory where the output CSV files will be saved (default is 'clean_data').
    """
    
    def __init__(self, factor_file, label_file, col_names, output_dir='clean_data'):
        """
        Initializes the DataSummaryGenerator with file paths and column names.
        
        Parameters:
        -----------
        factor_file : str
            Path to the CSV file containing factor data.
        label_file : str
            Path to the CSV file containing labels.
        col_names : list of str
            List of column names for the combined factor and label data.
        output_dir : str, optional
            Directory where the output CSV files will be saved (default is 'clean_data').
        """
        self.factor_file = factor_file
        self.label_file = label_file
        self.col_names = col_names
        self.output_dir = output_dir

    def load_data(self):
        """
        Loads the factor data and labels, concatenates them, and applies the column names.
        
        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame containing the concatenated factor data and labels with the specified column names.
        """
        # Load factor data and label data from CSV files
        factor_data = pd.read_csv(self.factor_file, index_col=0)
        label = pd.read_csv(self.label_file, index_col=0)

        # Concatenate the factor data and labels horizontally (i.e., along columns)
        factor_with_label = pd.concat([factor_data, label], axis=1)

        # Assign user-provided column names to the concatenated DataFrame
        factor_with_label.columns = self.col_names
        
        return factor_with_label

    def generate_summary(self):
        """
        Generates summary statistics (mean and standard deviation) grouped by the 'Group' column and saves them as CSV files.
        
        Outputs:
        --------
        mean.csv : CSV file
            The mean values of each factor by group.
        stdv.csv : CSV file
            The standard deviation of each factor by group.
        """
        # Load and prepare the concatenated factor and label data
        factor_with_label = self.load_data()

        # Group data by the 'Group' column and compute the mean for each group, then transpose the result
        mean_summary = factor_with_label.groupby(['Group']).mean().T

        # Group data by the 'Group' column and compute the standard deviation for each group, then transpose the result
        stdv_summary = factor_with_label.groupby(['Group']).std().T

        # Save the mean and standard deviation summaries to CSV files in the specified output directory
        mean_summary.to_csv(f'{self.output_dir}/mean.csv', index=True)
        stdv_summary.to_csv(f'{self.output_dir}/stdv.csv', index=True)

        print(f"Mean and standard deviation summaries have been saved to {self.output_dir}.")

# Example usage
if __name__ == "__main__":
    # List of column names, including the 'Group' column, provided by the user
    col_names = ['Momentum', 'Excess Return', 'Size', 'Value', 'Profitability', 'Investment', 
                 'Interest Rate', 'Inflation', 'Consumer Discretionary', 'Consumer Staples', 
                 'Energy', 'Financial', 'Health Care', 'Industrial', 'Materials', 'Technology', 
                 'Utilities', 'Group']

    # Create an instance of the DataSummaryGenerator class with file paths and column names
    summary_generator = DataSummaryGenerator(
        factor_file='clean_data/factor_returns.csv',  # Path to the factor data file
        label_file='clean_data/labels.csv',           # Path to the label data file
        col_names=col_names                           # List of column names for the combined data
    )

    # Generate the summary statistics and save them as CSV files
    summary_generator.generate_summary()
