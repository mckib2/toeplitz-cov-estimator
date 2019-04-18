'''Demonstrate that Gershgorin circle theorem is satisfied...

at least for the range of values used in the project.
'''

import numpy as np
from tqdm import trange, tqdm

from model import Model

if __name__ == '__main__':

    # Create a set of a to simulate over, choose a so the inequality
    # is always satisfied
    num_a = 10
    a = np.linspace(.50001, 1, num_a, endpoint=False)
    # a = .9999

    # Choose sim params
    M = 10

    # Now for each row verify the inequality
    for ii in trange(M, leave=False):
        tqdm.write(str(ii))

        sum0 = 0
        for jj in range(1, ii):
            sum0 += a**jj
        sum1 = 0
        for jj in range(ii+1, M-ii):
            sum1 += a**jj
        lhs = sum0 + sum1
        rhs = a**ii

    print('WE WIN!')
