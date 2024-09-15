"""
A Gaussian Mixture Model for clustering factor returns.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Optional


class GaussianMixtureModel:
    """A Gaussian Mixture Model for clustering factor returns."""

    def __init__(self, X: pd.DataFrame, n_components: Optional[int] = None):
        """Initialize the GMM model

        Args:
            X: array-like of shape (n_samples, n_features).
                The input data. Each row is a single sample.
            n_components: int, optional. Default=None.
                If provided, the model will be fitted with this number of components;
                otherwise, the optimal number of components is determined by the BIC score.
        """
        self.X = X

        if n_components is not None:
            self._n_components = n_components
        else:
            # Find the optimal number of components using BIC
            n_components_candidates = np.arange(1, 11)
            models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
                      for n in n_components_candidates]
            bics = [model.bic(X) for model in models]  # BIC for each model
            self._n_components = n_components_candidates[np.argmin(bics)]

        # Fit the model with the optimal number of components
        self.model = GaussianMixture(
            self._n_components,
            random_state=0,
            covariance_type='full'
        ).fit(X)

    def n_components(self) -> int:
        """Get the number of components in the mixture

        Returns:
            The number of components in the mixture.
        """
        return self._n_components

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the cluster for each sample in X

        Args:
            X: array-like of shape (n_samples, n_features).
                The input data. Each row is a single sample.

        Returns:
            labels: array of shape (n_samples,).
                The predicted cluster for each sample.
        """
        labels = self.model.predict(X)
        # Arrange the labels in a time series format
        return pd.Series(labels, index=self.X.index)

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
