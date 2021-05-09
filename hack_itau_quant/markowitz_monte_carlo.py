import matplotlib.pyplot as plt
import numpy as np


class MarkowitzMonteCarlo:
    
    def __init__(self, expected_returns, cov_matrix, n_portfolios = 10000):
        
        self._n_portfolios = n_portfolios
        self._n_assets = len(expected_returns)
        
        self._cov_matrix = cov_matrix
        self._expected_returns = expected_returns     
        
    def plot_efficient_frontier(self):
        
        self._simulate()
            
        plt.scatter(np.array(self.vols).reshape(-1), np.array(self.returns).reshape(-1))
        
    def get_min_vol(self):
        
        self._simulate()
        
        min_vol_index = np.argmin(self.vols)
        
        weights = np.array(self.weights[min_vol_index]).squeeze()
        
        return weights

        
    def _generate_random_weights(self):
        
        seed = np.random.random((self._n_assets, 1))
        
        weights = seed / np.sum(seed)
        
        return weights
    
    def _simulate(self):
        
        self.returns = []
        self.vols = []
        self.weights = []
        
        for i in range(self._n_portfolios):
            
            weights = self._generate_random_weights()
            
            rp = np.dot(weights.T, self._expected_returns)
            sigma_p = np.sqrt(np.dot(weights.T, np.dot(self._cov_matrix, weights)))
            
            self.returns.append(rp)
            self.vols.append(sigma_p)
            self.weights.append(weights)     
    