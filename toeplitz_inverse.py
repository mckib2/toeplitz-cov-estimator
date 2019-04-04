#pylint: ski[-file
'''Find closed form solution of Toeplitz inverse.'''

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import unit_impulse


if __name__ == '__main__':

    # Make a Toeplitz matrix
    N = 3
    theta = np.random.normal(5, 1, N)
    # theta = np.arange(N)
    T = toeplitz(theta)
    for ii in range(N):
        for jj in range(ii):
            T[ii, jj] = T[jj, ii]
    assert np.allclose(T, T.T)
    # print(T)

    # # Take first two columns
    # x = 1/T[:, 0]
    # y = 1/T[:, 1]
    d0 = unit_impulse(N)
    d1 = unit_impulse(N, 1)
    # print(d0)
    # print(d1)
    x = np.linalg.solve(T, d0)
    y = np.linalg.solve(T, d1)
    # print(x, y)

    # Construct T1
    T1 = toeplitz(y)
    for jj in range(N):
        for ii in range(jj):
            T1[ii, jj] = y[N-jj+ii]
    # print(T1)

    # Construct T2
    T2 = toeplitz(x)
    for jj in range(N):
        for ii in range(jj):
            T2[ii, jj] = x[N-jj+ii]
    # print(T2)

    # Construct U1
    U1 = np.eye(N)
    for jj in range(N):
        for ii in range(jj):
            U1[ii, jj] = x[N-jj+ii]
    # print(U1)

    # Construct U1
    U2 = np.zeros(U1.shape)
    for jj in range(N):
        for ii in range(jj):
            U2[ii, jj] = y[N-jj+ii]
    # print(U2)

    Tinv = (-T1.dot(U1) + T2.dot(U2))

    # print(Tinv)
    # print(np.linalg.inv(T))
    #
    # print(T.dot(np.linalg.inv(T)))
    print(T.dot(Tinv))
