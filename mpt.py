"""
An Implementation of the Modern Portfolio Theory (MPT) in Python.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco


class MarkowitzPortfolio:
    def __init__(self, mean_returns, cov_matrix, rf):
        """Initialize the class with mean returns, covariance matrix, and risk-free rate.

        Args:
            mean_returns (pd.Series): Expected returns for each asset.
            cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
            rf (float): Risk-free rate.
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.rf = rf

    def optimize(self):
        """
        Optimize the portfolio to maximize the Sharpe ratio.
        """
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix)

        # Objective function
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
            """Negative Sharpe ratio for optimization."""
            returns = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(returns - self.rf) / std

        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Bounds
        bound = (0.0, 1.0)
        bounds = tuple(bound for asset in range(num_assets))

        # Initial guess
        initial_guess = num_assets * [1. / num_assets]

        try:
            # Perform the optimization
            result = sco.minimize(neg_sharpe_ratio, initial_guess, args=args,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

        # Return the optimized weights
        return result.x


if __name__ == '__main__':
    # test a simple example
    mean_returns = pd.Series([0.08, 0.30, 0.3, 0.08])
    cov_matrix = pd.DataFrame([[0.01, 0.02, 0.015, 0.005],
                               [0.02, 0.04, 0.03, 0.01],
                               [0.015, 0.03, 0.025, 0.0075],
                               [0.005, 0.01, 0.0075, 0.0025]])
    rf = 0.05
    portfolio = MarkowitzPortfolio(mean_returns, cov_matrix, rf)
    print(portfolio.optimize())
