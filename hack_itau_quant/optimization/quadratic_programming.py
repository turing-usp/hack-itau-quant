import numpy as np

class QuadraticProgrammig:

    def __init__(self, B: np.array, A: np.array, c: np.array, b: np.array=None):
        """
        Args
        - B: Hessian Matrix
        - A: Restrictions
        - c: Ax = c
        - b : min x' * B * x - b' * x
        """
        self._n = B.shape[0]
        self._m = A.shape[0]

        self._B = B
        self._A = A
        self._c = c
        if b is None:
            self._b = np.zeros((self._n, 1))
        else:
            self._b = b

    def solve(self) -> np.array:
        C, d = self.combine_matrices()
        x = np.dot(np.linalg.inv(C), d)
        w, l = x[:self._n], x[self._n:]
        return w, l 

    def combine_matrices(self):
        zeros = np.zeros((self._m, self._m))
        left_side = np.concatenate((self._B, self._A), axis=0)
        right_side = np.concatenate((self._A.T, zeros), axis=0)
        C = np.concatenate((left_side, right_side), axis=1)
        d = np.concatenate((self._b, self._c), axis=0)
        return (C, d)
