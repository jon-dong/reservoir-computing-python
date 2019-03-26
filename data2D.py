"""
Generates 2D time series for Reservoir Computing.
Based on the Kuramoto-Sivashinsky differential equation
"""

import numpy as np
from sklearn.utils import check_random_state


def kuramoto_sivashinsky(sequence_length=1000, n_sequence=1,  spatial_points=100):
    '''
    solution of the Kuramoto–Sivashinsky equation, u_t + u*u_x + α*u_xx + γ*u_xxxx = 0,
    computed by tanh-function method.
    '''
    # Octave functions are download from https://github.com/qyxiao/machine-learning-2016-spring/blob/master
    from oct2py import octave
    N = spatial_points
    h = 0.25 # time step length
    nstp = sequence_length
    # a0 = np.zeros([N - 2, 1])
    L = 22.
    input_data = np.zeros((n_sequence, sequence_length+1, spatial_points+1))
    tt = np.zeros((n_sequence, 1, sequence_length+1))
    xx = np.zeros((n_sequence, spatial_points+1, 1))
    for idx in range(n_sequence):
        a0 = np.random.rand(N-2,1)/4  # just some initial condition
        [tt[idx], fdata] = octave.feval('ksfmstp', a0, L, h, nstp, 1, nout=2)
        [xx[idx], input_data[idx,:,:]] = octave.feval('ksfm2real', fdata, L, nout=2)

    # tanh function seems to help to the prediction accuracy
    return np.tanh(input_data), xx, tt