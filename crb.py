'''Get the CRB limit for our model.'''

import numpy as np

def getCRB(M, N, Ri):
    '''Cramer-Rao bound class for Toeplitz covariance matrix.

    Parameters
    ----------
    M : int
        Length of random vector, X.
    N : int
        Number of observations.
    Ri : array_like
        Inverse covariance matrix

    Returns
    -------
    '''

    block = np.eye(M)
    G = np.zeros((M**2, M))
    for ii in range(M):
        G[M*ii:M*(ii+1), :] = np.roll(block, (0, ii), axis=-1)

    J = M/2*G.T.dot(np.kron(Ri, Ri)).dot(G)

    return np.abs(1/J[:, 0])

if __name__ == '__main__':
    pass
