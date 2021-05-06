import numpy as np 

class Markowitz:

    def __init__(self, expected_returns, cov_matrix):
        self._n_assets = len(expected_returns)
        self._cov_matrix = cov_matrix
        self._expected_returns = expected_returns.reshape((self._n_assets, 1))

    def _build_useful_matrices(self):
        W = np.linalg.inv(self._cov_matrix)
        e = np.ones((self._n_assets, 1))

        We = W @ e
        Wm = W @ self._expected_returns

        return e, We, Wm

    def _build_magic_numbers(self):
        e, We, Wm = self._build_useful_matrices()

        A = e.T @ Wm
        B = self._expected_returns.T @ We
        C = e.T @ We
        D = B * C - A**2
        
        return A, B, C, D, We, Wm

    def _solve(self):
        A, B, C, D, We, Wm = self._build_magic_numbers()

        x_min = We / C
        z = Wm - A/C * We

        return x_min, z

    def optimal_risk(self, target_risk):
        x_min, z = self._solve()

        return x_min +  (z * target_risk / 2)

    def optimal_return(self, target_return):

        A, B, C, D, _, _ = self._build_magic_numbers()

        target_risk = 2 * (target_return - A / C) * C / D

        if target_risk < 0:
            raise ValueError("Target Return Not Valid!")

        return self.optimal_risk(target_risk)



