import numpy as np
import scipy.linalg

class WhiteningNormalizer(object):
    def __init__(self, controls, reg_param):
        # Whitening transform on population level data
        self.mu = np.mean(controls, axis=0)
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)
        
    def whitening_transform(self, X, lambda_, rotate=True):
        C = (1/X.shape[0]) * np.dot(X.T, X)
        s, V = scipy.linalg.eigh(C)
        D = np.diag( 1. / np.sqrt(s + lambda_) )
        W = np.dot(V, D)
        if rotate:
            W = np.dot(W, V.T)
        self.W = W

    def normalize(self, X):
        return np.dot(X - self.mu, self.W)
