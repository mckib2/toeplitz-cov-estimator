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
    array_like
        The Cramer-Rao bound.

    Notes
    -----
    Note that this should not be flat.  You have fewer entries for
    Toeplitz entries further down the column, so lower SNR for these
    lower elements!
    '''

    # We can construct the Fisher information matrix by finding where
    # each parameter is repeated in the structured covariance matrix
    G = np.zeros((M**2, M))
    ones = np.ones(M)
    for ii in range(M):
        upper = np.diag(ones, ii)[:M, :M].astype(bool)
        lower = np.diag(ones, -ii)[:M, :M].astype(bool)
        block = (upper | lower).astype(float)
        G[:, ii] = block.flatten()
    # print(G)

    # block = np.eye(M)
    # G = np.zeros((M**2, M))
    # for ii in range(M):
    #     G[M*ii:M*(ii+1), :] = np.roll(block, (0, ii), axis=-1)
    # print(G)

    # print(Ri)
    J = N/2*G.T.dot(np.kron(Ri, Ri)).dot(G)
    return np.abs(1/J[:, 0]) # Should this ever be negative?

if __name__ == '__main__':
    pass
