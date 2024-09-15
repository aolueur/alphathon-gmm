import pandas as pd
import numpy as np
from model import GaussianMixtureModel


# Load the data
data = pd.read_csv('./clean_data/factor_returns.csv', index_col=0)

# Create an instance of the GaussianMixtureModel
gmm = GaussianMixtureModel(data)

# Result
print(gmm.predict(data))


