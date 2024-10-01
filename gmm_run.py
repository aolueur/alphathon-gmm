import pandas as pd
from model import GaussianMixtureModel
from utils import principal_component_analysis, visualize_gmm_results


# Load the data
data = pd.read_csv('./clean_data/factor_log_returns.csv', index_col=0)

# Perform PCA
data_pca = principal_component_analysis(data, n_components=3)

# Create an instance of the GaussianMixtureModel
gmm = GaussianMixtureModel(data_pca)

# Result
print(gmm.n_components())
predictions = gmm.predict(data_pca)
predictions.name = 'Group'
predictions.index.name = 'Date'

predictions.to_csv('./clean_data/labels.csv')
predictions.to_json('./clean_data/labels.json')

visualize_gmm_results(predictions)
