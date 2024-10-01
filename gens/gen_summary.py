import sys
from utils import DataSummaryGenerator


# Example usage
if __name__ == "__main__":
    # List of column names, including the 'Group' column, provided by the user
    col_names = ['Momentum', 'Excess Return', 'Size', 'Value', 'Profitability', 'Investment',
                 'Interest Rate', 'Inflation', 'Consumer Discretionary', 'Consumer Staples',
                 'Energy', 'Financial', 'Health Care', 'Industrial', 'Materials', 'Technology',
                 'Utilities', 'Group']

    # List of factors to include in covariance/correlation matrix calculations
    selected_factors = ['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financial',
                        'Health Care', 'Industrial', 'Materials', 'Technology', 'Utilities']

    # Create an instance of the DataSummaryGenerator class with file paths and column names
    summary_generator = DataSummaryGenerator(
        factor_file='clean_data/factor_returns.csv',  # Path to the factor data file
        label_file='clean_data/labels.csv',           # Path to the label data file
        # List of column names for the combined data
        col_names=col_names,
        # List of specific factors to include
        selected_factors=selected_factors
    )

    # Generate the covariance and correlation matrices by group (only for the selected factors)
    summary_generator.generate_summary()
    summary_generator.generate_cov_corr_by_group()
