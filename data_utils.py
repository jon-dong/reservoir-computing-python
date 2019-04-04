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
    - input_data: input sequence, np array (n_sequence, sequence_length, 1)
    - y: output sequence, np array (n_sequence, sequence_length, 1)
"""

import numpy as np
from sklearn.utils import check_random_state


def roll_and_concat(input_data, roll_num=1):
    '''
    :param input_data: the original data that will be rolled by axis1
    :param roll_num: how many times will be the original data rolled and concatenated with itself
    '''
    n_sequence, sequence_len, input_dim = input_data.shape
    rolled_data = np.zeros((n_sequence, sequence_len, input_dim*roll_num))
    for i_roll in range(roll_num):
        rolled_data[:, :, i_roll*input_dim:(i_roll+1)*input_dim] = np.roll(input_data, -i_roll-1, axis=1)

    return rolled_data


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


def get_kuramoto_sivashinsky_lyap_exp(sequence_length=1000, n_sequence = 100, spatial_points=64):
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