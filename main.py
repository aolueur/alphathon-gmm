import pandas as pd
from model import GaussianMixtureModel
from utils import principal_component_analysis


# Load the data
data = pd.read_csv('./clean_data/factor_returns.csv', index_col=0)

# Perform PCA
data_pca = principal_component_analysis(data, n_components=3)

# Create an instance of the GaussianMixtureModel
gmm = GaussianMixtureModel(data_pca)

# Result
print(gmm.n_components())
print(gmm.predict(data_pca))
