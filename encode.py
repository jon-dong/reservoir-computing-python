"""
Encoding functions, to transform a vector into another format using elementwise operations.

These functions are generic and can be used outside the Reservoir Computing framework.
"""

import numpy as np


def phase_encoding(mat, scaling_factor=np.pi, n_levels=None):
    """ Transforms a real-valued vector into a phase-only vector """
    mat = mat * scaling_factor
    if n_levels:
        step = 2 * np.pi / n_levels
        mat = np.round(mat / step) * step
    return np.exp(1j * mat)


def binary_threshold(mat, threshold):
    """ A simple threshold function """
    return mat > threshold


def naive_binary(mat, lower_bound=-0.5, higher_bound=0.5, binary_dim=10):
    """ We generate a binary vector using a series of equally-spaced thresholds """
    if mat.ndim == 1:  # If the matrix is a vector
        mat = mat[..., np.newaxis]  # Transform into a matrix
    step = (higher_bound - lower_bound) / binary_dim

    enc_input_data = np.repeat(np.zeros(mat.shape), binary_dim, axis=-1)
    for i_binary in range(binary_dim):
        enc_input_data[..., i_binary::binary_dim] = mat > lower_bound + step * i_binary
    return enc_input_data


def local_binary(mat, lower_bound=-0.5, higher_bound=0.5, step=1, binary_dim=10):
    if mat.ndim == 1:  # If the matrix is a vector
        mat = mat[..., np.newaxis]  # Transform into a matrix
    normalized_mat = (mat - lower_bound) / (higher_bound - lower_bound)
    step = np.random.uniform(0, step, binary_dim)

    enc_input_data = np.repeat(np.zeros(mat.shape), binary_dim, axis=-1)
    for i_binary in range(binary_dim):
        enc_input_data[..., i_binary::binary_dim] = \
            np.mod((normalized_mat - 2*step[i_binary]*np.random.uniform()) // step[i_binary], 2) == 0
    return enc_input_data


def large_bin_binary(mat, lower_bound=-0.5, higher_bound=0.5, binary_dim=10, gamma=1, balanced=True):
    if mat.ndim == 1:  # If the matrix is a vector
        mat = mat[..., np.newaxis]  # Transform into a matrix
    normalized_mat = (mat - lower_bound) / (higher_bound - lower_bound)
    # step = (binary_dim - 3) / (4 * binary_dim)
    if balanced:
        factor = np.floor(gamma * binary_dim) * 2 - 1
        step = 1/(4/factor*binary_dim-2)
        dither_vec = np.arange(-step, 1+step, (1+2*step)/binary_dim)
    else:
        factor = np.floor(gamma * binary_dim) * 2 - 1
        step = 1/(4/factor*binary_dim)
        dither_vec = np.arange(1/(2*binary_dim), 1, 1/binary_dim)

    enc_input_data = np.repeat(np.zeros(mat.shape), binary_dim, axis=-1)
    for i_binary in range(binary_dim):
        if 0 < i_binary < binary_dim:
            enc_input_data[..., i_binary::binary_dim] = \
                abs(normalized_mat - dither_vec[i_binary]) < step
        elif i_binary == 0:
            enc_input_data[..., i_binary::binary_dim] = \
                normalized_mat - dither_vec[i_binary] < step
        elif i_binary == binary_dim:
            enc_input_data[..., i_binary::binary_dim] = \
                normalized_mat - dither_vec[i_binary] > -step
    return enc_input_data


def fixed_binary(mat, lower_bound=-0.5, higher_bound=0.5, binary_dim=12):
    if mat.ndim == 1:  # If the matrix is a vector
        mat = mat[..., np.newaxis]  # Transform into a matrix
    normalized_mat = (mat - lower_bound) / (higher_bound - lower_bound)
    step = [2/3, 1/3] * int(binary_dim / 2)
    dither_vec = np.arange(0.333, 1.001, 0.667/binary_dim)
    step = [2/3, 1/3, 2/9] * int(binary_dim / 3)
    dither_vec = np.arange(0.333, 1.001, 0.667/binary_dim)
    # step = [2/3, 1/3, 2/9, 1/6] * int(binary_dim / 4)
    # dither_vec = np.arange(0.333, 1.001, 0.667/binary_dim)

    enc_input_data = np.repeat(np.zeros(mat.shape), binary_dim, axis=-1)
    for i_binary in range(binary_dim):
        enc_input_data[..., i_binary::binary_dim] = \
            np.mod((normalized_mat - dither_vec[i_binary]) // step[i_binary], 2) == 0
    return enc_input_data


def bit_encoding(mat, lower_bound=-0.5, higher_bound=0.5, binary_dim=8):
    if mat.ndim == 1:
        mat = mat[..., np.newaxis]
    normalized_mat = (mat - lower_bound) / (higher_bound - lower_bound)
    step = 1 / 2**binary_dim
    bit_mat = np.floor(normalized_mat / step)

    enc_input_data = np.repeat(np.zeros(mat.shape), binary_dim, axis=-1)
    for i_bit in range(binary_dim):
        enc_input_data[..., i_bit::binary_dim] = \
            np.mod(np.floor(bit_mat / 2**i_bit), 2) == 1
    return enc_input_data
