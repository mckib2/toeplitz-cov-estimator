'''Demonstrate CRBs using exponential model and random.'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from crb import getCRB
from model import Model

if __name__ == '__main__':

    # Make a model using exponential and random
    N = 10
    M = 10
    niter = 10000

    CRB0 = np.zeros((niter, M))
    CRB1 = np.zeros((niter, M))
    for ii in trange(niter, leave=False):
        X0 = Model(M, exp=True)
        X1 = Model(M)

        # Compute Cramer-Rao bounds
        CRB0[ii, :] = getCRB(M, N, np.linalg.inv(X0.R))
        CRB1[ii, :] = getCRB(M, N, np.linalg.inv(X1.R))

    # Take a look
    plt.plot(np.mean(CRB0, axis=0), label='Exponential model')
    plt.plot(np.mean(CRB1, axis=0), label='Random')
    plt.title('Average CRB Comparison')
    plt.xlabel('parameters')
    plt.ylabel('Error variance')
    plt.legend()
    plt.show()
