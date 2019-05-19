"""
Generates 2D time series for Reservoir Computing.
Based on the Kuramoto-Sivashinsky differential equation
"""

import numpy as np
import random
import scipy.io


def kuramoto_sivashinsky(sequence_length=1000, n_sequence=1,  spatial_points=100):
    '''
    solution of the Kuramoto–Sivashinsky equation, u_t + u*u_x + α*u_xx + γ*u_xxxx = 0,
    computed by tanh-function method.
    '''
    # Octave functions are downloaded from https://github.com/qyxiao/machine-learning-2016-spring/blob/master
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
    # we remove first time and spatial indices because sometimes they get bad values
    input_data = input_data[:,1:,1:]
    tt = np.linspace(np.min(tt[:, :, :-1]), np.max(tt[:, :, :-1]), sequence_length)
    xx = xx[0, :-1, 0]
    return input_data, xx, tt

def kuramoto_sivashinsky_matlab(sequence_length=1000, n_sequence=1,  spatial_points=100):
    '''
    solution of the Kuramoto–Sivashinsky equation, u_t + u*u_x + α*u_xx + γ*u_xxxx = 0,
    computed by tanh-function method.
    '''
    # Matlab functions are downloaded from https://github.com/qyxiao/machine-learning-2016-spring/blob/master
    import matlab.engine
    matlab_eng = matlab.engine.start_matlab()
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
        [tt_, fdata] = matlab_eng.ksfmstp(matlab.double(a0.tolist()), L, h, nstp, 1, nargout=2)
        tt[idx] = np.array(tt_._data).reshape(tt_.size[::-1]).T
        [xx_, input_data_] = matlab_eng.ksfm2real(fdata, L, nargout=2)
        xx[idx] = np.array(xx_._data).reshape(xx_.size[::-1]).T
        input_data[idx,:,:] = np.array(input_data_._data).reshape(input_data_.size[::-1]).T
    matlab_eng.quit()
    # we remove first time and spatial indices because sometimes they get bad values
    input_data = input_data[:,1:,1:]
    tt = np.linspace(np.min(tt[:, :, :-1]), np.max(tt[:, :, :-1]), sequence_length)
    xx = xx[0, :-1, 0]
    return input_data, xx, tt

def kuramoto_sivashinsky_from_dataset(sequence_length=10000, n_sequence=1):
    spatial_points = 65
    if sequence_length <= 10000:
        i_seq = random.randint(0, 100 - n_sequence)
        j_seq = random.randint(0, 10000 - sequence_length)
        ks_data = np.load('KS_data/ks_data.npy')[range(i_seq, i_seq + n_sequence), j_seq:j_seq + sequence_length, :]
    else:
        ks_data = np.zeros((n_sequence, sequence_length, spatial_points))
        n_concat = int(sequence_length / 10000)
        sequence_length = sequence_length % 10000
        i_seq = random.randint(0, 100 - n_sequence - n_concat - 1)
        j_seq = random.randint(0, 10000 - sequence_length)
        for n in range(n_concat):
            ks_data[:, n * 10000:(n + 1) * 10000, :] = np.load('1D_kuramoto_sivashinsky_datasets/ks_data.npy')[
                                                       range(n + i_seq, n + i_seq + n_sequence), :-1, :]
        ks_data[:, -sequence_length:, :] = np.load('1D_kuramoto_sivashinsky_datasets/ks_data.npy')[
                                           range(n_concat + i_seq + 1, n_concat + i_seq + 1 + n_sequence),
                                           :sequence_length, :]
    x_axis = np.load('1D_kuramoto_sivashinsky_datasets/x_axis.npy')
    time_axis = np.load('1D_kuramoto_sivashinsky_datasets/time_axis.npy')
    time_max = time_axis[-1]/10001 * sequence_length
    time_axis = np.linspace(0, time_max, sequence_length)
    return ks_data, x_axis, time_axis
