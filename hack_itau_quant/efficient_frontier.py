import numpy as np
import matplotlib.pyplot as plt
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

    def plot_efficient_frontier(self, step_size, n_steps = 100):

        returns, risks = self._solver.get_efficient_curve(step_size = step_size, n_steps = n_steps)

        plt.plot(risks, returns)

        plt.xlabel('Volatilities')
        plt.ylabel('Returns')
        plt.title('Efficient Frontier')

        plt.show()

        return (returns, risks)

    def max_loss(self, loss, period, z_alpha = -1.645):

        start_return = self._solver.get_start_return()

        last_w = None
        for r in np.arange(start_return, 1, 0.00005):

            w = self.efficient_return(r).reshape(-1)
            rp = np.dot(w.T, self._expected_returns) * period
            sigma_p = np.sqrt(np.dot(w.T, np.dot(self._cov_matrix * np.sqrt(period), w)))
            
            z =  (loss - rp) / (sigma_p / np.sqrt(period))

            is_alternative_hypotesis = z <= z_alpha
            if is_alternative_hypotesis:
                last_w = w
            else:
                return last_w
