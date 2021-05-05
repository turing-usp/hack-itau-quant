import numpy as np

class EfficientFrontier:

    def __init__(self, mu: np.array, sigma: np.array):
        self._mu = mu
        self._sigma = sigma
