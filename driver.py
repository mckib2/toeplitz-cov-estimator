'''Driver script.'''

import warnings
from multiprocessing import Pool
from time import time, ctime
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Model
from crb import getCRB
from estimators import (meaninator, conventional, constrained_psd,
                        shrinkage, lasso)
from structured_estimators import CRZ
from utils import nearestPSD

def iter_fun(_ii, X, estimators, N, M):
    '''Picklable function that runs every iteration of parallel loop.
    '''

    # Sample the distribution
    Xs = X.sample(N)

    # Setup error array to send back (order not guaranteed)
    err = np.zeros((len(estimators), M))

    # Do estimation!
    for idx, key in enumerate(estimators):
        try:
            # err[idx] = compare_mse(X.R, estimators[key](Xs))
            err[idx, :] = X.R[0, :] - estimators[key](Xs)[0, :]
        except ValueError:
            # LASSO wants a few samples to work with
            pass
        # except UserWarning:
        #     # OAS will complain sometimes about only 1 sample
        #     pass
        # except RuntimeWarning:
        #     # LASSO also complains a lot
        #     pass

    return err

def mean_convential(xs):
    '''Picklable meaninator+conventional'''
    return meaninator(conventional(xs))

def mean_shrinkage(xs):
    '''Picklable meaninator+shrinkage'''
    return meaninator(shrinkage(xs))

def mean_lasso(xs):
    '''Picklable meaninator+lasso'''
    return meaninator(lasso(xs))

def nearestPSD_wrapper_conventional(xs):
    '''Wrapper for nearest PSD function.'''
    return nearestPSD(meaninator(conventional(xs)))

def nearestPSD_wrapper_shrinkage(xs):
    '''Wrapper for nearest PSD function.'''
    return nearestPSD(meaninator(shrinkage(xs)))

if __name__ == '__main__':

    #1: Model parameters and initialization
    M = 10
    N = 3
    X = Model(M)
    assert X.is_R_PSD() # Sanity check

    # # Is the precision matrix sparse?
    # plt.imshow(np.linalg.inv(X.R))
    # plt.show()

    # 2: Get the CRB estimates of theta
    CRB = getCRB(M, N, np.linalg.inv(X.R))

    # # 4.5. Estimate using conventional sample covariance estimator
    # Ns = [10, 100, 1000, 10000]
    # for N in Ns:
    #     xs = x.sample(N)
    #     Rhat0 = conventional(xs)
    #     Rhat1 = conventional(xs, biased=False)
    #     print('N: %d' % N)
    #     print('Rhat0 err: %g' % np.linalg.norm(x.R - Rhat0))
    #     print('Rhat1 err: %g' % np.linalg.norm(x.R - Rhat1))
    #     print('')

    # 5. Calculate the sample estimation error variance for your
    # estimator using a large number of Monte Carlo trials

    # WARNING: LASSO will take a long time!
    maxiter = 10000
    estimators = {
        'Sample': conventional,
        # 'Meaninator': mean_convential,
        # 'PSD-constrained': constrained_psd,
        # 'OAS': shrinkage,
        # 'OAS (mean)': mean_shrinkage,
        # 'GraphicalLassoCV': lasso,
        # 'GraphicalLassoCV (mean)': mean_lasso,
        'CRZ': CRZ,
        # 'nearestPSD (sample)': nearestPSD_wrapper_conventional,
        'nearestPSD (OAS)': nearestPSD_wrapper_shrinkage
    }

    # LASSO,, OAS gonna whine, so let's make ignore
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Let's parallelize this bad boy
    chunksize = 10
    piter = partial(iter_fun, X=X, estimators=estimators, N=N, M=M)
    t0 = time() # start the timer
    with Pool() as pool:
        res = list(tqdm(pool.imap(piter, range(maxiter), chunksize),
                        leave=False, total=maxiter))
    err = np.array(res)
    print(err.shape)

    # Print output
    print('ITERS: %d' % maxiter)
    print('    M: %d' % M)
    print('    N: %d' % N)
    longest = sorted([len(key) for key in estimators])[-1]
    print('AVG STATS:')
    print('%sMEAN%s\tSTD' % (' '*(longest+2), ' '*8))
    for ii, key in enumerate(estimators):
        print('%s%s: %f,\t%f' % (' '*(longest-len(key)), key,
                                 np.mean(err[ii, :]),
                                 np.std(err[ii, :])))

    # For posterity...
    np.savez(
        'results/N%d_M%d_niter%d_t%s' % (N, M, maxiter, ctime()),
        X.R, N, M, maxiter, CRB, list(estimators.keys()), err)

    # See how we did
    plt.plot(CRB, '.-', label='CRB')
    for ii, key in enumerate(estimators):
        plt.plot(np.std(err[ii, :]**2, axis=0), '.--', label=key)
    plt.legend()
    plt.show()
