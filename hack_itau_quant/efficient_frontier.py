import numpy as np
from .optimization import QuadraticProgrammig


class EfficientFrontier:

    def __init__(self, expected_returns: np.array, cov_matrix: np.array):
        self._expected_returns = expected_returns
        self._cov_matrix = cov_matrix

        self._n_assets = len(expected_returns)

    def _build_restrictions(self, target_return):

        weight_restriction = np.ones(self._n_assets)

        A = np.array([weight_restriction, self._expected_returns])
        c = np.array([1, target_return]).reshape(2, 1)

        return A, c

    def get_wallet(self, target_return: float):

        A, c = self._build_restrictions(target_return)

        qp = QuadraticProgrammig(B = 2 * self._cov_matrix, A = A, c = c)

        w, l = qp.solve()

        return w
