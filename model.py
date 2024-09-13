"""
A Gaussian Mixture Model for clustering factor returns.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import yfinance as yf
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike


class GaussianMixtureModel:
    """A Gaussian Mixture Model for clustering factor returns.

    Attributes:
        X: array-like of shape (n_samples, n_features).
            The input data. Each row is a single sample.
        n_components: int.
            The number of components (Gaussian distributions) in the mixture.
            The optimal number of components is determined by the BIC score.
        model: GaussianMixture.
            The fitted Gaussian Mixture model
    """

    def __init__(self, X: ArrayLike):
        """Initialize the GMM model

        Args:
            n_components: int.
                The number of components (Gaussian distributions) in the mixture.
            X: array-like of shape (n_samples, n_features).
                The input data. Each row is a single sample.
        """
        self.X = X

        # Find the optimal number of components
        n_components_candidates = np.arange(1, 21)
        models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
                  for n in n_components_candidates]
        bics = [model.bic(X) for model in models]  # BIC for each model
        self.n_components = n_components_candidates[np.argmin(bics)]

        # Fit the model with the optimal number of components
        self.model = GaussianMixture(
            self.n_components,
            random_state=0,
            covariance_type='full'
        ).fit(X)

    def n_components(self) -> int:
        """Get the number of components in the mixture

        Returns:
            n_components: int.
                The number of components in the mixture.
        """
        return self.n_components

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


def plot_aic_bic(X: ArrayLike):
    """Helper function for plotting the AIC and BIC scores for different number of components

    Args:
    X: array-like of shape (n_samples, n_features). 
        The input data. Each row is a single sample.
    """

    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(
        X) for n in n_components]

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criteria')
    plt.title('AIC and BIC Scores for Gaussian Mixture Models')
    plt.show()


if __name__ == '__main__':
    # Fetch the factor returns
    data = yf.download(
        'AAPL MSFT GOOGL AMZN',
        period='3mo'
    )['Adj Close'].pct_change().dropna() * 100

    gmm = GaussianMixtureModel(data)
    print(gmm.n_components)
    print(gmm.predict(data))

    plot_aic_bic(data)
