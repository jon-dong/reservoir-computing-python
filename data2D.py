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


def kuramoto_sivashinsky_lyap_exp(sequence_length=500, n_sequence = 500, spatial_points=20):
    '''
    Lyapunov exponent for the Kuramoto–Sivashinsky equation, u_t + u*u_x + α*u_xx + γ*u_xxxx = 0.
    The method is adapted and modified from "Physica D 65 (1993) 117-134" paper, also from
    https://blog.abhranil.net/2014/07/22/calculating-the-lyapunov-exponent-of-a-time-series-with-python-code/
    '''
    from oct2py import octave
    N = spatial_points
    h = 0.25 # time step length
    nstp = sequence_length
    L = 22.
    dlist = [[] for i in range(sequence_length)]
    for i0 in range(n_sequence):
        a0 = np.random.rand(N-2,1)/2  # just some initial condition
        [_, fdata] = octave.feval('ksfmstp', a0, L, h, nstp, 1, nout=2)
        [_, input_data0] = octave.feval('ksfm2real', fdata, L, nout=2)

        a1 = a0 + 10**(-7)
        [_, fdata] = octave.feval('ksfmstp', a1, L, h, nstp, 1, nout=2)
        [_, input_data1] = octave.feval('ksfm2real', fdata, L, nout=2)

        for k in range(sequence_length):
            dlist[k].append(np.log(np.linalg.norm(input_data0[k,:]-input_data1[k,:])))

    spectrum = np.zeros((sequence_length))
    for i in range(sequence_length):
        spectrum[i] = sum(dlist[i]) / len(dlist[i])

    lyap_exp = np.polyfit(range(len(spectrum)), spectrum, 1)[0]/h

    return lyap_exp