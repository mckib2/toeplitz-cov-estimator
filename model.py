'''Gaussian random variables with Toeplitz covariance matrix.'''

import numpy as np
from scipy.linalg import toeplitz
from tqdm import trange

class Model(object):
    '''M length Gaussian random vector.

    Attributes
    ----------
    M : int
        Lenth of random vector.
    R : array_like, optional
        Covariance matrix, Toeplitz.
    '''

    def __init__(self, M, R=None, exp=False, a=.9):
        '''Initialize model for vector of random variables.

        Parameters
        ----------
        M : int
            Length of random vector.
        R : array_like, optional
            Covariance matrix.
        exp : bool, optional
            Whether or not to generate R using exponential model or
            choose random parameters until a positive semidefinite,
            Toeplitz R is found.
        a : float, optional
            If using exponential model, the parameter a. 0 < a < 1.
        '''
        self.M = M

        if R is None:

            # Geometric model satisfies Gershgorin circle theorem,
            # but only works well when a close to 1...
            if exp:
                col = np.zeros(M)
                for ii in range(M):
                    col[ii] = a**ii
                self.R = toeplitz(col)

            else:

                # Make a Toeplitz covariance matrix, guess random cols
                # until we find a PSD one.  Not elegant, but works
                # with some reliablity. Sorting really helps speed it
                # up.
                self.R = np.diag(np.ones(M)*-1) + 1
                cnt = 0
                while not self.is_R_PSD():
                    self.R = toeplitz(
                        np.sort(np.random.normal(0, 1, M))[::-1])
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

    def is_R_PSD(self, tol=0):
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
        tol = np.abs(tol)
        E = np.linalg.eigvals(self.R)
        return np.all(E > -tol)

if __name__ == '__main__':

    # Test model to make sure we get PSD toeplitz matrices
    M = 10
    err = 0
    for _ii in trange(1000, leave=False):
        X = Model(M)
        try:
            assert X.is_R_PSD()
        except AssertionError:
            err += 1
    print('%d errors' % err)
