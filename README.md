# GMM for Alphathon 2024
GMM Model for Alphathon

## Model

The `GaussianMixtureModel` class is a wrapper around the `sklearn.mixture.GaussianMixture` class. It provides a simple interface to train and predict using a Gaussian Mixture Model.

The `GaussianMixtureModel` constructor takes
- `data`: The training data to be fed into the model. The data should be a 2D numpy array with the shape `(n_samples, n_features)`. The `n_samples` is the number of samples in the dataset and `n_features` is the number of features in each sample.and an optional parameter `n_components` (default None).  
- `n_components` (optional): The number of components in the GMM model. If not specified, the model will choose the optimal number of components using the Bayesian Information Criterion (BIC).
  
The `GaussianMixtureModel` class has the following methods:
- `n_components(self)`: Returns the number of components in the GMM model.
- `predict(self, data)`: Predicts the cluster labels for the given data. The data should be a 2D numpy array with the shape `(n_samples, n_features)`. Returns a 1D numpy array containing the predicted cluster labels.
- `weights(self)`: Returns the weights of the components in the GMM model.
- `means(self)`: Returns the means of the components in the GMM model.
- `covariances(self)`: Returns the covariances of the components in the GMM model.
- `bic(self)`: Returns the Bayesian Information Criterion (BIC) of the GMM model.
- `aic(self)`: Returns the Akaike Information Criterion (AIC) of the GMM model.