"""
A Gaussian Mixture Model for clustering factor returns.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import yfinance as yf

from numpy.typing import ArrayLike


class GaussianMixtureModel:
    """
    A Gaussian Mixture Model for clustering factor returns.
    """

    def __init__(self, n_components: int, X: ArrayLike):
        """Initialize the GMM model

        Args:
            n_components: int. 
                The number of components (Gaussian distributions) in the mixture.
            X: array-like of shape (n_samples, n_features). 
                The input data. Each row is a single sample.
        """
        self.n_components = n_components
        self.X = X
        self.model = GaussianMixture(
            n_components,
            random_state=0,
            covariance_type='full'
        ).fit(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the cluster for each sample in X

        Args:
            X: array-like of shape (n_samples, n_features). 
                The input data. Each row is a single sample.

        Returns:
            labels: array of shape (n_samples,). 
                The predicted cluster for each sample.
        """
        return self.model.predict(X)

    def weights(self) -> np.ndarray:
        """Get the weights of the Gaussian distributions

        Returns:
            weights: array of shape (n_components, ). 
                The weights of each mixture component.
        """
        return self.model.weights_

    def means(self) -> np.ndarray:
        """Get the means of the Gaussian distributions

        Returns:
            means: array of shape (n_components, ). 
                The means of each mixture component.
        """
        return self.model.means_

    def covariances(self) -> np.ndarray:
        """Get the covariances of the Gaussian distributions

        Returns:
            covariances: array of shape (n_components, n_features, n_features). 
                The covariances of each mixture component.
        """
        return self.model.covariances_

    def aic(self):
        """Compute the Akaike Information Criterion (AIC)

        Returns:
            aic: float.
                The AIC for the model. The lower the better
        """
        return self.model.aic(self.X)

    def bic(self):
        """Compute the Bayesian Information Criterion (BIC)

        Returns:
            bic: float.
                The BIC for the model. The lower the better
        """
        return self.model.bic(self.X)


if __name__ == '__main__':
    # Fetch the factor returns
    data = yf.download(
        'AAPL MSFT GOOGL AMZN',
        period='3mo'
    )['Adj Close'].pct_change().dropna() * 100

    gmm = GaussianMixtureModel(4, data)
    print(gmm.predict(data))
