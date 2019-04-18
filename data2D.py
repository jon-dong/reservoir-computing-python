"""
Generates 2D time series for Reservoir Computing.
Based on the Kuramoto-Sivashinsky differential equation
"""

import numpy as np
import random


def convection_dataset(sequence_length=10000, n_sequence=1):
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
