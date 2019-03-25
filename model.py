'''Gaussian random variables with Toeplitz covariance matrix.'''

import numpy as np
from scipy.linalg import toeplitz, eigh

class Model(object):
    '''M length Gaussian random vector.

    Attributes
    ----------
    M : int
        Lenth of random vector.
    R : array_like
        Covariance matrix, Toeplitz.
    '''

    def __init__(self, M, R=None):
        '''Initialize model for vector of random variables.

        Parameters
        ----------
        M : int
            Length of random vector.
        R : array_like, optional
            Covariance matrix.
        '''
        self.M = M

        if R is None:
            # Make a Toeplitz covariance matrix, guess random ones
            # until we find a PSD one.  Not elegant, but works with
            # some reliablity.
            self.R = np.diag(np.ones(M)*-1) + 1
            cnt = 0
            while not self.is_R_PSD():
                self.R = toeplitz(np.random.normal(0, 1, M))
                cnt += 1
            # print('took %d times' % cnt)
        else:
            self.R = R

    def sample(self, N):
        '''Observe N independent samples of X.

        Parameters
        ----------
        N : int
            Number of independent samples to observe.

        Returns
        -------
        array_like
            N samples of the multivariate normal distribution.
        '''
        m = np.zeros(self.M)
        return np.random.multivariate_normal(m, self.R, int(N))

    def is_R_PSD(self, tol=1e-8):
        '''Check to see if R is positive semi-definite.

        Parameters
        ----------
        tol : float, optional
            The smallest allowable negative eigenvalue.

        Returns
        -------
        bool
            Whether or not R is positive semi-definite.
        '''
        E, _V = eigh(self.R)
        return np.all(E > -tol)
