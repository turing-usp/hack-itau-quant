import numpy as np
from .optimization import Markowitz

class EfficientFrontier:

    def __init__(self, expected_returns: np.array, cov_matrix: np.array):
        self._expected_returns = expected_returns.to_numpy()
        self._cov_matrix = cov_matrix.to_numpy()
        self._solver = Markowitz(
            expected_returns = self._expected_returns, 
            cov_matrix = self._cov_matrix)

    def efficient_risk(self, target_risk):
        return self._solver.optimal_risk(target_risk)

    def efficient_return(self, target_return):
        return self._solver.optimal_return(target_return)

