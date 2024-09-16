import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture

from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
