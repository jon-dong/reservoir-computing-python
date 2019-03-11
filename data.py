"""
Generates data for Reservoir Computing.

List of functions:
    - memory: predict a previous input of a random input
    - xor: compute the xor on two consecutive previous inputs of a binary input
    - narma: chaotic time series

Typical parameters:
    - sequence_length=1000: length of the time series
    - memory_delay=1: how many time steps in the past for the current task
    - random_state=None: possibility of a random seed for reproducible results

Output:
    - input_data: input sequence, np array (sequence_length, 1)
    - y: output sequence, np array (sequence_length, 1)
"""

import numpy as np
from sklearn.utils import check_random_state


def memory(sequence_length=1000, n_sequence=1, memory_delay=1, random_state=None):
    '''
    Simple memory task:
    Given a time series, can we recall what was the value memory_delay in the past?
    '''
    random_state = check_random_state(random_state)

    input_data = random_state.normal(loc=0., scale=1, size=(n_sequence, sequence_length, 1))
    y = np.roll(input_data, memory_delay, axis=1)

    return input_data, y


def xor(sequence_length=1000, n_sequence=1, memory_delay=1, random_state=None):
    '''
    XOR time series
    '''
    random_state = check_random_state(random_state)

    input_data = random_state.randint(low=0, high=2, size=(n_sequence, sequence_length, 1))
    shifted_input_data = np.roll(input_data, 1, axis=1)
    output_data = np.logical_xor(input_data, shifted_input_data)
    # Additional shift by memory_delay
    y = np.roll(output_data, memory_delay, axis=1)

    return input_data, y


def narma(sequence_length=1000, n_sequence=1, random_state=None):
    '''
    NARMA time series, canonical test in Reservoir Computing
    '''
    random_state = check_random_state(random_state)

    input_data = random_state.uniform(high=0.5, size=(n_sequence, sequence_length, 1))
    y = np.zeros((n_sequence, sequence_length))

    # Recursive NARMA equation for y
    for i in range(10, sequence_length):
        y[:, i] = 0.3 * y[:, i-1] + \
            0.05 * y[:, i-1] * np.prod(y[:, i-10:i-1]) + \
            1.5 * input_data[:, i-10] * input_data[:, i-1] + 0.1

    return np.tanh(input_data), np.tanh(y)


def mackey_glass(sequence_length=1000, n_sequence=1, data_dim=1, random_state=None):
    '''
    Generates Mackey-Glass time series, using discretization of the 
    MG differential equation
    '''
    random_state = check_random_state(random_state)

    # Parameters of the differential equation
    beta = 0.2  # weight of the non-linear term with memory
    gamma = 0.1  # exponential decay constant
    tau = 17  # memory
    n = 10  # power in the non-linear term
    h = 1  # time step
    memory_length = int(tau/h)

    # The dataset is calculated for longer length, then the additional part will be cut from begining of dataset.
    # This is done because the beginning of the dataset is always spoiled
    add_to_sec = round(sequence_length/10)
    sequence_length = sequence_length + add_to_sec
    input_data = np.zeros((n_sequence, sequence_length, data_dim))  # last dimension is input_dim = 1
    # Initialization of Mackey Glass
    input_data[:, :memory_length] = 1.1 + 0.2 * random_state.normal(loc=0., scale=1, size=(n_sequence, memory_length, 1))
    # Computation of next terms by finite differences
    for iSequence in range(memory_length, sequence_length):
        input_data[:, iSequence, 0] = (1 - h * gamma) * input_data[:, iSequence - 1, 0] + \
        beta * h * input_data[:, iSequence - memory_length, 0] / \
        (1 + input_data[:, iSequence - memory_length, 0] ** n)

    # Preprocessing (done by other people, seems to help)
    input_data = np.tanh(input_data - 1)[:, add_to_sec:, :]

    return input_data

def mso(sequence_length=1000, n_sequence=1, random_state=None):
    '''
    Generate the Multiple Sinewave Oscillator time-series, a sum of two sines
    with incommensurable periods. 
    '''
    random_state = check_random_state(random_state)

    input_data = np.zeros((n_sequence, sequence_length, 1))  # last dimension is input_dim = 1
    for i_sequence in range(n_sequence):
        phase = np.random.rand()
        x = np.atleast_2d(np.arange(sequence_length)).T
        input_data[i_sequence, :] = np.sin(0.2 * x + phase) + np.sin(0.311 * x + phase)

    y = np.roll(input_data, -1, axis=1)
    return input_data, y

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


def roll_and_concat(input_data, roll_num=1):
    '''
    :param input_data: the original data that will be rolled by axis1
    :param roll_num: how many times will be the original data rolled and concatenated with itself
    '''
    n_sequence, sequence_length, spatial_points = input_data.shape
    rolled_data = np.zeros((n_sequence, sequence_length, spatial_points * roll_num))
    for i in range(roll_num):
        rolled_data[:, :, i * spatial_points:(i + 1) * spatial_points] = np.roll(input_data, -(i + 1), axis=1)

    return rolled_data