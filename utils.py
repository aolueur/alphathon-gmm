import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture

from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Sector ETF Name -> Ticker mapping
to_ticker = {
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Financial': 'XLF',
    'Health Care': 'XLV',
    'Industrial': 'XLI',
    'Technology': 'XLK',
    'Materials': 'XLB',
    'Utilities': 'XLU'
}

# Sector ETF Ticker -> Name mapping
to_name = {
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLF': 'Financial',
    'XLV': 'Health Care',
    'XLI': 'Industrial',
    'XLK': 'Technology',
    'XLB': 'Materials',
    'XLU': 'Utilities'
}


class TimeSeriesDataHandler:
    def __init__(self, filepath_return: list[str], filepath_price: list[str]):
        """Initialize the TimeSeriesDataHandler

        Attributes:
            filepath_return: list of str.
                List of file paths for the return data.
            filepath_price: list of str.
                List of file paths for the price data.
            datasets: list of pd.DataFrame.
                List to store the datasets
        """
        self.filepath_return = filepath_return
        self.filepath_price = filepath_price
        self.datasets = []

    def load_return_data(self):
        """Load the return data from the input files"""
        for filepath in self.filepath_return:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, format='%Y%m%d')
            data.index = data.index.strftime('%Y-%m-%d')
            self.datasets.append(data)

    def load_price_data(self):
        """Load the price data from the input files"""
        for filepath in self.filepath_price:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, format='%Y%m%d')
            data.index = data.index.strftime('%Y-%m-%d')
            data = (np.log(data) - np.log(data.shift(1))) * 100
            self.datasets.append(data)

    def clean_data(self):
        """Clean the data by replacing missing values and filling forward"""
        for data in self.datasets:
            data.replace([-99.99, -999], pd.NA, inplace=True)
            data.ffill(inplace=True)

    def filter_before_date(self, date: str):
        """Filter data before a given date, yyyymmdd"""
        for i in range(len(self.datasets)):
            self.datasets[i] = self.datasets[i][self.datasets[i].index >= date]

    def download_ticker_data(
        self, ticker: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Download ticker data using yfinance

        Args:
            ticker: list of str.
                List of the ticker symbols. For example, ['AAPL', 'MSFT'].
            start_date: str.
                The start date in the format 'YYYY-MM-DD'.
            end_date: str.
                The end date in the format 'YYYY-MM-DD'."""

        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        ticker_data.index = ticker_data.index.strftime('%Y-%m-%d')
        close = ticker_data['Adj Close']
        return (np.log(close) - np.log(close.shift(1))) * 100

    def add_ticker_data(
        self, tickers: list[str], start_date: str, end_date: str
    ):
        """Add ticker data to the datasets

        Args:
            ticker: str.
                List of the ticker symbols. For example, ['AAPL', 'MSFT'].
            start_date: str.
                The start date in the format 'YYYY-MM-DD'.
            end_date: str.
                The end date in the format 'YYYY-MM-DD'."""
        ticker_data = self.download_ticker_data(tickers, start_date, end_date)
        for ticker in tickers:
            if ticker in ticker_data.columns:
                self.datasets.append(ticker_data[ticker].rename(ticker))

    def merged_datasets(self) -> pd.DataFrame:
        """Merge the datasets into a single DataFrame"""
        merged_data = self.datasets[0]
        for data in self.datasets[1:]:
            merged_data = pd.merge(
                merged_data,
                data,
                left_index=True,
                right_index=True,
                how='left',
            )
        merged_data.ffill(inplace=True)
        return merged_data[19:-1]


def plot_aic_bic(X: ArrayLike):
    """Helper function for plotting the AIC and BIC scores for different number of components

    Args:
    X: array-like of shape (n_samples, n_features).
        The input data. Each row is a single sample.
    """

    n_components = np.arange(1, 11)
    models = [
        GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
        for n in n_components
    ]

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criteria')
    plt.title('AIC and BIC Scores for Gaussian Mixture Models')
    plt.show()


def principal_component_analysis(data: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """ Perform Principal Component Analysis on the input data

    Args:
        data: array-like of shape (n_samples, n_features).
            The input data. Each row is a single sample.

    Returns:
        array-like of shape (n_samples, n_features).
            The transformed data.
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components)
    data_pca = pca.fit_transform(data_scaled)

    # Return the transformed data
    return pd.DataFrame(data_pca, index=data.index)


def visualize_gmm_results(data: pd.Series):
    """
    Plots the most frequent classes for each month divided into three subplots based on GMM predictions.

    Args:
        data: The predicted cluster for each sample.
    """
    # Prepare the data
    prediction = pd.DataFrame(data).reset_index().rename(
        columns={'index': 'Date', 0: 'Group'})

    # Convert the 'date' column to datetime if it's not already
    prediction['Date'] = pd.to_datetime(prediction['Date'])

    # Extract year and month for grouping
    prediction['Year_Month'] = prediction['Date'].dt.to_period('M')

    # Determine the most frequent class for each month
    most_frequent_class = prediction.groupby('Year_Month')['Group'].apply(
        lambda x: Counter(x).most_common(1)[0][0])

    # Convert to a DataFrame for plotting
    classification_result = most_frequent_class.reset_index(
        name='Most_Frequent_Class')

    # Mapping classes to colors for visualization
    class_colors = {
        0: '#58C9EF',  # Cyan
        1: '#F3D403',  # Yellow
        2: '#5E2D79',  # Purple
        3: '#E94B3C',  # Red
        4: '#34D399',  # Green
        5: '#FFA500'   # Orange
    }

    # Convert 'Year_Month' to timestamp for plotting
    classification_result['Year_Month'] = classification_result['Year_Month'].dt.to_timestamp()

    # Split the data into three parts
    split_date1 = pd.Timestamp('2008-01-01')
    split_date2 = pd.Timestamp('2016-01-01')

    df_part1 = classification_result[classification_result['Year_Month'] < split_date1]
    df_part2 = classification_result[(classification_result['Year_Month'] >= split_date1) & (
        classification_result['Year_Month'] < split_date2)]
    df_part3 = classification_result[classification_result['Year_Month'] >= split_date2]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharey=True)

    # Plot the first part
    ax1.bar(df_part1['Year_Month'], height=1, color=[class_colors[cls]
            for cls in df_part1['Most_Frequent_Class']], width=25, edgecolor='none')
    ax1.set_title('Most Frequent Class per Month (1999-2007)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Class')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])

    # Plot the second part
    ax2.bar(df_part2['Year_Month'], height=1, color=[class_colors[cls]
            for cls in df_part2['Most_Frequent_Class']], width=25, edgecolor='none')
    ax2.set_title('Most Frequent Class per Month (2008-2015)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Class')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])

    # Plot the third part
    ax3.bar(df_part3['Year_Month'], height=1, color=[class_colors[cls]
            for cls in df_part3['Most_Frequent_Class']], width=25, edgecolor='none')
    ax3.set_title('Most Frequent Class per Month (2016-2024)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Class')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])

    # Create a common legend
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4, label=label)
                      for label, color in class_colors.items()]
    fig.legend(handles=legend_patches, title='Classes',
               bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2)

    plt.tight_layout()
    plt.show()


def generate_data(filepath_return: list[str], filepath_price: list[str]):
    """Generate the data to be fed into the GMM model

    Args:
        filepath_return: list of str.
            List of file paths for the return data.
        filepath_price: list of str.
            List of file paths for the price data.
    """
    # Create an instance of the TimeSeriesDataHandler
    ts_data_handler = TimeSeriesDataHandler(filepath_return, filepath_price)

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
    df_normal_returns = (np.exp(df/100) - 1)*100
    with open('./clean_data/factor_log_returns.csv', 'w') as csv_file:
        df.to_csv(path_or_buf=csv_file)
    with open('./clean_data/factor_returns.csv', 'w') as csv_file:
        df_normal_returns.to_csv(path_or_buf=csv_file)


class DataSummaryGenerator:
    """
    A class to generate summary statistics (mean, standard deviation, covariance, and correlation)
    for factor data with labels, and save the results to CSV files by group.
    """

    def __init__(self, factor_file, label_file, col_names, output_dir='clean_data', selected_factors=None):
        """
        Initializes the DataSummaryGenerator with file paths and column names.

        Parameters:
        -----------
        factor_file : str
            Path to the CSV file containing the factor data.
        label_file : str
            Path to the CSV file containing the labels.
        col_names : list of str
            List of column names for the combined factor and label data.
        output_dir : str, optional
            Directory where the output CSV files will be saved (default is 'clean_data').
        selected_factors : list of str, optional
            List of specific factor columns to include in covariance/correlation calculations.
            If not provided, all factors will be used.
        """
        self.factor_file = factor_file
        self.label_file = label_file
        self.col_names = col_names
        self.output_dir = output_dir
        # Default to all except 'Group'
        self.selected_factors = selected_factors if selected_factors else col_names[:-1]

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

        print(
            f"Mean and standard deviation summaries have been saved to {self.output_dir}.")

    def generate_cov_corr_by_group(self):
        """
        Generates covariance and correlation matrices for selected factors in each group and saves them as separate CSV files.

        Outputs:
        --------
        covariance/covariance_Group.csv : CSV file for each group
            The covariance matrix for the selected factors in each group.

        correlation/correlation_Group.csv : CSV file for each group
            The correlation matrix for the selected factors in each group.
        """
        # Load the data
        factor_with_label = self.load_data()

        # Get the unique groups
        groups = factor_with_label['Group'].unique()

        for group in groups:
            # Extract data for the current group, and select only the relevant factors
            group_data = factor_with_label[factor_with_label['Group']
                                           == group][self.selected_factors]

            # Compute the covariance matrix
            covariance_matrix = group_data.cov()

            # Compute the correlation matrix
            correlation_matrix = group_data.corr()

            # Save the covariance and correlation matrices as separate CSV files
            covariance_matrix.to_csv(
                f'{self.output_dir}/covariance_{group}.csv')
            correlation_matrix.to_csv(
                f'{self.output_dir}/correlation_{group}.csv')

        print(
            f"Covariance and correlation matrices for each group (based on selected factors) have been saved to {self.output_dir}/covariance and {self.output_dir}/correlation.")


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
            raise ValueError(
                "Data not loaded. Call load_data() before saving merged data.")

        # Perform inner join and left join
        combined_partial = pd.merge(
            self.labels, self.data, on='Date', how='inner')
        combined_full = pd.merge(self.labels, self.data, on='Date', how='left')
        combined_full = combined_full[combined_full['Date']
                                      >= combined_partial['Date'][0]]
        combined_full.ffill(inplace=True)

        # Save the merged data to CSV files
        combined_partial.to_csv(output_file_partial, index=False)
        combined_full.to_csv(output_file_full, index=False)

        print(f"Partial merged dataset saved to {output_file_partial}")
        print(f"Full merged dataset saved to {output_file_full}")
