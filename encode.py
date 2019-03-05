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


def local_binary(mat, lower_bound=-0.5, higher_bound=0.5, step=0.5, binary_dim=10, enc_init=False, step_vec=None, dither_vec=None):
    if mat.ndim == 1:  # If the matrix is a vector
        mat = mat[..., np.newaxis]  # Transform into a matrix
    normalized_mat = (mat - lower_bound) / (higher_bound - lower_bound)
    if enc_init == False:
        step_vec = np.random.uniform(step/2, step, binary_dim)
        dither_vec = np.random.uniform(0, 1, binary_dim)

    enc_input_data = np.repeat(np.zeros(mat.shape), binary_dim, axis=-1)
    for i_binary in range(binary_dim):
        enc_input_data[..., i_binary::binary_dim] = \
            np.mod((normalized_mat - dither_vec[i_binary]) // step_vec[i_binary], 2) == 0
    
    return enc_input_data, step_vec, dither_vec
