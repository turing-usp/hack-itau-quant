import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

class Denoising:
    
    def __init__(self, returns: pd.DataFrame, n_facts: int, alpha: float, b_width: float = .01):
        
        self._returns = returns
        self._n_facts = n_facts
        self._alpha = alpha
        self._b_width = b_width
        self._n_points = returns.shape[0]
        
        self._q = returns.shape[0] / float(returns.shape[1])
        
        self.cov_matrix = np.corrcoef(returns, rowvar = 0)
        
        self.e_val, self.e_vec = self._get_pca()
        
    def remove_noise_with_mean(self):
        
        max_eval, variance = self._find_max_eval()
        
        n_facts = self.e_val.shape[0] - np.diag(self.e_val)[::-1].searchsorted(max_eval)
        
        corr_matrix = self._filter_mean(n_facts)
        
        return corr_matrix
    
    def remove_noise_with_shrinkage(self):
        
        max_eval, variance = self._find_max_eval()
        
        n_facts = self.e_val.shape[0] - np.diag(self.e_val)[::-1].searchsorted(max_eval)
        
        corr_matrix = self._filter_shrinkage(self.e_val, self.e_vec, n_facts, alpha = self._alpha)
        
        return corr_matrix
        
        
    def _filter_mean(self, n_facts):
        """Substitui ruído pela média
        """

        e_val = np.diag(self.e_val).copy()

        e_val[n_facts:] = e_val[n_facts:].sum()/float(e_val.shape[0] - n_facts) 
        e_val = np.diag(e_val)

        cov = np.dot(self.e_vec, e_val).dot(self.e_vec.T)
        corr = Denoising.cov2corr(cov)

        return corr
    
    def _filter_shrinkage(self, eVal, eVec, nFacts, alpha = 0):

        e_val_left  = self.e_val[:n_facts, :n_facts]
        e_vec_left = self.e_vec[:, :n_facts] 
        e_val_right, e_vec_right = self.e_val[n_facts:, n_facts:], self.e_vec[:, n_facts:]
        
        corr_left = np.dot(e_vec_left, e_val_left).dot(e_vec_left.T)
        corr_right = np.dot(e_vec_right, e_val_right).dot(e_vec_right.T)
        
        corr_shrinkage = corr_left + alpha * corr_right + (1 - alpha) * np.diag(np.diag(corr_right))
        
        return corr_shrinkage

    def _get_pca(self):
        """
        Function that gets eigenvalues and eigenvectors
        of a given Hermitian matrix
        
        Args:
            - matrix (pd.DataFrame or np.array)
        Returns:
            - (tuple) eigenvalues (matrix), eigenvectors (matrix)
        """
        e_val, e_vec = np.linalg.eigh(self.cov_matrix)

        indices = e_val.argsort()[::-1]

        e_val, e_vec = e_val[indices], e_vec[:,indices]
        e_val = np.diagflat(e_val)
        
        return e_val, e_vec


    def _find_max_eval(self):
        
        out = minimize(lambda var: self._compare_theoretical_and_empirical(var),
                       x0=np.array(0.5), bounds=((1E-5, 1-1E-5),))

        if out['success']: var = out['x'][0]
        else: var=1
            
        e_max = var*(1+(1./self._q)**.5)**2
        
        return e_max, var
    
    def _compare_theoretical_and_empirical(self, var):
                
        theoretical_pdf = Denoising.marcenko_pastur_pdf(var[0], self._q, self._n_points) 
        empirical_pdf = Denoising.fit_kde(np.diag(self.e_val), self._b_width, x = theoretical_pdf.index.values)
        
        mean_squared_errors = np.sum((empirical_pdf - theoretical_pdf)**2)
        
        return mean_squared_errors

    @staticmethod
    def fit_kde( obs, b_width = .25, kernel = "gaussian", x = None):
        """Fit kernel to a series of obs, and derive the prob of obs
        Args:
            - obs (np.array): array with eingenvalues sorted desc
            - bWidth (float): 
            - kernel (str): type of kernel to apply on KDE method
            - x (np.array or None): array of values on which the fit KDE will be evaluated
        Returns:
            - (pd.Series) estimated density function
        """
        
        if len(obs.shape)==1:
            obs = obs.reshape(-1,1)
            
        kde = KernelDensity(kernel = kernel, bandwidth = b_width).fit(obs)
        
        if x is None: 
            x = np.unique(obs)
            
        if len(x.shape) == 1: 
            x = x.reshape(-1,1)
        
        logProb = kde.score_samples(x) # log(density)
        pdf = pd.Series(np.exp(logProb), index = x.flatten())
        
        return pdf
    
    @staticmethod
    def marcenko_pastur_pdf(var, q, pts):
        """Generates random variable with Marcenko-pastur
        distribuition
        Args:
            - var (float): variance
            - q (float): T/N, where T and N are the dimensions of a matrix
            - pts (int): number of points to generate
        Returns: 
            - (pd.Series) generated random variables
        """
        # eMin and eMax are the minimum and maximum eigenvalues
        eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
        # eVal is an array with length of pts between eMin and eMax
        eVal = np.linspace(eMin, eMax, pts)
        # calculates probability function for eVal
        pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
        # converts to pd.Series
        pdf = pd.Series(pdf, index = eVal)
        return pdf 
    
    @staticmethod
    def cov2corr(cov):
        # Derive the correlation matrix from a covariance matrix
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[corr < -1] ,corr[corr > 1] = -1, 1 # numerical error
        return corr