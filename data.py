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

To-do:
    - Add other time series from the Matlab Code

Created: 2017/11/22 by Jonathan Dong
Last modified: 2017/11/23 by Jonathan Dong
"""

import numpy as np
from sklearn.utils import check_random_state


def memory(sequence_length=1000, memory_delay=1, random_state=None):
    random_state = check_random_state(random_state)
    input_data = random_state.normal(loc=0., scale=1, size=(sequence_length, 1))

    y = np.roll(input_data, memory_delay)

    return input_data, y


def xor(sequence_length=100, memory_delay=1, random_state=None):
    random_state = check_random_state(random_state)
    input_data = random_state.randint(low=0, high=2, size=(sequence_length, 1))

    shifted_input_data = np.roll(input_data, 1)
    output_data = np.logical_xor(input_data, shifted_input_data)
    y = np.roll(output_data, memory_delay)

    return input_data, y


def narma(sequence_length=1000, random_state=None):
    random_state = check_random_state(random_state)
    input_data = random_state.uniform(high=0.5, size=(sequence_length, 1))
    y = np.zeros((sequence_length, 1))

    for i in range(10, sequence_length):
        y[i] = 0.3 * y[i-1] + \
            0.05 * y[i-1] * np.prod(y[i-10:i-1]) + \
            1.5 * input_data[i-10] * input_data[i-1] + 0.1

    return np.tanh(input_data), np.tanh(y)
