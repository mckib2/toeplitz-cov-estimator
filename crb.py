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
        Inverse covariance matrix (precision matrix).

    Returns
    -------
    array_like
        The Cramer-Rao bound for desired parameter vector.

    Notes
    -----
    Note that this should not be flat.  You have fewer entries for
    Toeplitz entries further down the column, so fewer samples for
    these lower elements (see construction of Jacobian, G).
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

    J = G.T.dot(np.kron(Ri, Ri)).dot(G)
    assert np.all(np.linalg.eigvals(J) >= 0)
    C = 1/N*np.linalg.inv(J)
    assert np.all(np.diag(C) >= 0)
    return np.diag(C)

if __name__ == '__main__':

    # Sanity checks for computing CRB of Toeplitz model
    from model import Model
    from tqdm import trange
    M = 10
    N = 10
    X = Model(M)

    for _ii in trange(1000, leave=False):
        # Find the CRB
        CRB = getCRB(M, N, np.linalg.inv(X.R))
        assert np.all(CRB >= 0)
