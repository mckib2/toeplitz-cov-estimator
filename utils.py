'''Utilities for project 1.'''

# import numpy as np
#
# def nearestPSD(A, epsilon=0):
#     '''Find nearest positive-semi-definite matrix to A.
#
#     Notes
#     -----
#     Non-iterative approach. This is slightly modified from Rebonato
#     and Jackel (1999) (page 7-9).
#     '''
#     n = A.shape[0]
#     eigval, eigvec = np.linalg.eig(A)
#     val = np.matrix(np.maximum(eigval, epsilon))
#     vec = np.matrix(eigvec)
#     T = 1/(np.multiply(vec, vec) * val.T)
#     T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
#     B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
#     out = B*B.T
#     return out

import numpy as np

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearestPSD(A, nit=10):
    '''Find nearest positive-semi-definite matrix to A.

    Notes
    -----
    Iterative approach from Higham (2000).
    '''
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity
    # matrix here) the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for _k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk
