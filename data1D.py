"""
Generates 1D time series for Reservoir Computing.
List of functions:
    - mackey_glass: Mackey Glass differential equation
    - mso: Multiple Sinewave Oscillator
Typical parameters:
    - sequence_length=1000: length of the time series
    - memory_delay=1: how many time steps in the past for the current task
    - random_state=None: possibility of a random seed for reproducible results
Output:
    - input_data: input sequence, np array (n_sequence, sequence_length, 1)
The output for a prediction task typically is: y = np.roll(input_data, -delay, axis=1)
"""

import numpy as np
from sklearn.utils import check_random_state


def mackey_glass(sequence_length=1000, n_sequence=1, random_state=None):
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

    input_data = np.zeros((n_sequence, sequence_length+memory_length, 1))  # last dimension is input_dim = 1
    # Initialization of Mackey Glass
    input_data[:, :memory_length] = 1.1 + 0.2 * random_state.normal(loc=0., scale=1, size=(n_sequence, memory_length, 1))
    # Computation of next terms by finite differences
    for iSequence in range(memory_length, sequence_length+memory_length):
        input_data[:, iSequence, 0] = (1 - h * gamma) * input_data[:, iSequence - 1, 0] + \
        beta * h * input_data[:, iSequence - memory_length, 0] / \
        (1 + input_data[:, iSequence - memory_length, 0] ** n)

    # Preprocessing (done by other people, seems to help)
    input_data = np.tanh(input_data[:, memory_length:] - 1)
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

    return input_data