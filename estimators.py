'''Estimator to find Toeplitz covariance matrix.'''

from functools import partial

import numpy as np
# import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
# from scipy.signal import periodogram, welch
from scipy.optimize import minimize
from sklearn.covariance import oas, GraphicalLassoCV

def lasso(xs):
    '''Use GraphicalLassoCV.

    Parameters
    ----------
    xs : array_like
        N samples of X.

    Returns
    -------
    C : array_like
        Covariance matrix estimate.

    Notes
    -----
    This implementation uses cross-validation to find the correct
    weight for Lasso.  Graphical Lasso is a method for finding a
    sparse inverse covariance matrix, so this is additional
    information that might not follow from having a Toeplitz
    covariance matrix...
    '''
    model = GraphicalLassoCV(cv=3)
    model.fit(xs)
    C = model.covariance_
    return C

def shrinkage(xs):
    '''Estimate covariance using Oracle Approximating shrinkage.

    Parameters
    ----------
    xs : array_like
        N samples of X.

    Returns
    -------
    C : array_like
        Covariance matrix estimation.
    '''
    C, _alpha = oas(xs, assume_centered=True)
    return C

def meaninator(C):
    '''The meaninator: use mean along diagonals of given covariance.

    Parameters
    ----------
    C : array_like
        Covariance estimate.

    Returns
    -------
    Rhat : array_like
        Estimate of covariance matrix R.
    '''
    M = C.shape[0]

    # Make Toeplitz by using mean along all diagonals
    Rhat = np.zeros(C.shape)
    rows, cols = np.indices(C.shape)
    for kk in range(-M+1, M):
        val = np.mean(np.diag(C, kk))
        r = np.diag(rows, kk)
        c = np.diag(cols, kk)
        Rhat[r, c] = val

    # We can also check to see if it's symmetric
    assert np.allclose(Rhat, Rhat.T), 'Estimate should be symmetric!'

    return Rhat

def ordinator(_xs):
    '''Make use of sparsity of ordered samples to get estimate.'''

def obj(x0, ref_Pxx):
    '''Objective for constrained_psd optimization.

    Parameters
    ----------
    x0 : array_like
        M length vector of diagonal values for Toeplitz covariance
        matrix.
    ref_Pxx : array_like
        Estimate of power spectral density for the sample covariance
        matrix.

    Returns
    -------
    float
        l2 norm between PSDs.
    '''
    R = toeplitz(x0)
    Pxx = np.fft.fft(R, axis=0)**2
    return np.linalg.norm(ref_Pxx - Pxx)

def constrained_psd(xs):
    '''Optimize Toeplitz covariance to match PSD of observed samples.

    Parameters
    ----------
    xs : array_like
        N samples of X.

    Returns
    -------
    array_like
        Estimate of Toeplitz covariance matrix.
    '''

    # Need to find out how to do better PSD straight from samples
    # ref_Pxx = np.fft.fft(conventional(xs), axis=0)**2
    ref_Pxx = np.fft.fft(shrinkage(xs), axis=0)**2
    # _f, ref_Pxx = welch(
    #     xs, return_onesided=False, scaling='spectrum', axis=0)

    pobj = partial(obj, ref_Pxx=ref_Pxx)
    x0 = meaninator(conventional(xs))[:, 0]
    res = minimize(pobj, x0)
    return toeplitz(res['x'])


def conventional(xs, biased=True):
    '''Estimate using sample covariance estimator.

    Parameters
    ----------
    xs : array_like
        N samples of X.
    biased : bool
        Use biased or unbiased formula.

    Returns
    -------
    Rhat : array_like
        Estimate of covariance matrix R.
    '''

    # Numpy implementation:
    # xs0 = xs - np.average(xs, axis=0)[None, :]
    xs0 = xs.copy() # we know zero mean
    fact = xs0.shape[0] - int(not biased)
    C = np.dot(xs0.T, xs0)/fact

    # Naive implementation
    # xs0 = xs - np.average(xs, axis=0)[None, :]
    # fact = xs0.shape[0] - int(not biased)
    # C = np.zeros((xs0.shape[1], xs0.shape[1]))
    # for ii in range(xs0.shape[0]):
    #     C += np.outer(xs0[ii, :], xs0[ii, :])
    # C /= fact

    # # Numpy has a built in one, so check to make sure we did the
    # # right thing
    # ref = np.cov(xs, rowvar=False, bias=biased)
    # assert np.allclose(C, ref)
    return C
