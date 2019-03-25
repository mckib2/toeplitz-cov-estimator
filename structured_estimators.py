'''Estimators with Toeplitz in mind from the get-go.'''

import numpy as np
from estimators import conventional, meaninator

def CRZ(xs):
    '''Cai-Ren-Zhou approach for optimal estimator.

    Parameters
    ----------
    xs : array_like
        N samples of M-dimensional Gaussian normal distributed RV, X.

    Returns
    -------
    array_like
        Toeplitz covariance estimate.

    Notes
    -----
    See: http://www.stat.yale.edu/~hz68/Toeplitz.pdf

    This seems to do well for M > N, no so great for N > M.
    '''

    _N, M = xs.shape[:]

    # We need to find weights, wm
    k = int(M/2)
    if np.mod(k, 2) > 0:
        k -= 1

    w = np.zeros(M)
    for m in range(M):
        if m <= k/2:
            w[m] = 1
        elif k/2 < m <= k:
            w[m] = 2 - 2*m/k
        else:
            w[m] = 0

    # We'll need the averaged sample covariance estimate, take the
    # first column:
    Cest = meaninator(conventional(xs))[:, 0]

    # Now construct the tapering estimator
    C = np.zeros((M, M))
    for idx in np.ndindex(C.shape):
        s, t = idx[:]
        C[s, t] = w[np.abs(s - t)]*Cest[np.abs(s - t)]

    return C
