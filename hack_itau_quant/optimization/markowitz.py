import numpy as np


class Markowitz:

    def __init__(self, expected_returns, cov_matrix):
        self._n_assets = len(expected_returns)
        self._cov_matrix = cov_matrix
        self._expected_returns = expected_returns.reshape((self._n_assets, 1))

        A, B, C, D, We, Wm = self._build_magic_numbers()

        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._We = We
        self._Wm = Wm

    def optimal_risk(self, target_risk):
        x_min, z = self._solve()

        weights = x_min + (z * target_risk / 2)

        if self._check_weights(weights):
            return weights
        else:
            raise ValueError("Target Risk Does Not Converge")

    def optimal_return(self, target_return):

        target_risk = 2 * (target_return - self._A /
                           self._C) * self._C / self._D

        if target_risk < 0:
            raise ValueError("Target Return Not Valid!")

        return self.optimal_risk(target_risk)

    def get_efficient_curve(self, step_size, n_steps):

        start_return = self._A / self._C

        end_return = start_return + step_size * n_steps

        returns = np.arange(start_return, end_return, step_size)

        risks = np.array([self._optimal_curve(target_return)
                          for target_return in returns])

        return (returns, risks)

    def _check_weights(self, weights):

        for w in weights:

            if w < 0:
                return False

        return True

    def _build_useful_matrices(self):
        W = np.linalg.inv(self._cov_matrix)
        e = np.ones((self._n_assets, 1))

        We = np.dot(W, e)
        Wm = np.dot(W, self._expected_returns)

        return e, We, Wm

    def _build_magic_numbers(self):
        e, We, Wm = self._build_useful_matrices()

        A = np.dot(e.T, Wm)
        B = np.dot(self._expected_returns.T, Wm)
        C = np.dot(e.T, We)
        D = B * C - A**2

        return A, B, C, D, We, Wm

    def _solve(self):

        x_min = self._We / self._C
        z = self._Wm - (self._A/self._C) * self._We

        return x_min, z

    def _optimal_curve(self, target_return):

        hyperbola = self._C * target_return ** 2 - 2 * self._A * target_return + self._B

        risk = np.sqrt((1 / self._D) * hyperbola)

        return float(risk)
