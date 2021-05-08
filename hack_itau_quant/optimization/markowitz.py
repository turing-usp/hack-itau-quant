import numpy as np


class Markowitz:
    """Solver of Markowitz optimization problem
    """

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
        """Calculate optimization solution given target risk
        Args:
            - target_risk (float): risk tolerance
        Returns:
            - (np.array) optimal weights
        """
        x_min, z = self._solve()

        weights = x_min + (z * target_risk / 2)

        return weights

    def optimal_return(self, target_return):
        """Calculate optimization solution given target return
        Args:
            - target_return (float): target return
        Returns:
            - (np.array) optimal weights
        """

        target_risk = 2 * (target_return - self._A /
                           self._C) * self._C / self._D

        return self.optimal_risk(target_risk)

    def get_efficient_curve(self, n_points):
        """Generates risk return curve with optimal weights solution
        Args:
            - n_points (int): number of points
        Return:
            - Tuple(np.array, np.array, np.array) returns, risks and optimal weights, respectively
        """

        start_return = self._A / self._C
        end_return = self._expected_returns.max()
        step_size = (end_return - start_return)/n_points

        returns = np.arange(start_return, end_return, step_size)
        weights = np.array([self.optimal_return(target_return) for target_return in returns])
        weights = weights.squeeze()

        risks = np.sqrt(np.array([np.dot(w.T, np.dot(self._cov_matrix, w)) for w in weights]))

        return (returns, risks, weights)

    def get_start_return(self):
        """Get initial return value of efficient curve
        """

        return self._A / self._C

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
