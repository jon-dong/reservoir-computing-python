"""
Reservoir class, a custom sklearn estimator for Reservoir Computing.

Methods:
    - fit: iterates the reservoir with training input, fits the output weights
    - predict: iterates the reservoir with input, predicts using the fit output weights

Internal methods:
    - initialize: initializes all reservoir internal weights
    - reset: resets the state of the reservoir
    - encode: encodes the input
    - activation: sets the activation function of the reservoir
    - iterate: iterates the reservoir with input, outputs the collected reservoir states
    - train: performs a non-linear regression to find the output weights
    - output: computes a predicted output given reservoir state and output weights

Parameters:
    - n_input: dimension of the input
    - n_res: size of the reservoir
    - input_scale: scale of the input weights
    - res_scale: scale of the res_scale (after normalization by 1/np.sqrt(n_res))
    - random_projection: random_projection type ('simulation', 'experiment')
    - weights_type: distribution of input and reservoir weights
    - encoding_method: encoding technique applied on the input fed to the reservoir
    - encoding_param: TBD
    - activation_fun: activation function for every unit of the reservoir
    - activation_param: TBD
    - forget: number of initial states we discard to get rid of transient states
    - train_method: algorithm for the output regression
    - train_param: TBD
    - random_state: random seed for reproducible results

Internal attributes:
    - state: current state of the reservoir, np array (n_res,)
    - input_w: input weights, np array (n_res, n_input)
    - res_w: reservoir internal weights, np array (n_res, n_res)
    - output_w: output weights, np array (n_res, 1)

To-do:
    -
"""

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from scipy.linalg import lstsq


class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self, n_input=1, n_res=100, input_scale=1, res_scale=1,
                 random_projection='simulation', weights_type='gaussian',
                 encoding_method=None, encoding_param=None,
                 activation_fun='tanh', activation_param=None, forget=100,
                 train_method='explicit', train_param=None,
                 random_state=None):
        self.n_input = n_input
        self.n_res = n_res
        self.input_scale = input_scale
        self.res_scale = res_scale
        self.random_projection = random_projection
        self.weights_type = weights_type
        self.encoding_method = encoding_method
        self.encoding_param = encoding_param
        self.activation_fun = activation_fun
        self.activation_param = activation_param
        self.forget = forget
        self.train_method = train_method
        self.train_param = train_param
        self.random_state = random_state

        self.state = None
        self.input_w = None
        self.res_w = None
        self.output_w = None

    def initialize(self):
        """ Initializes the reservoir state, the input and reservoir weights """
        self.random_state = check_random_state(self.random_state)
        self.state = self.random_state.normal(loc=0., scale=1, size=(self.n_res,))

        if self.random_projection == 'simulation':
            if self.weights_type == 'gaussian':
                self.input_w = self.random_state.normal(loc=0., scale=self.input_scale, size=(self.n_res, self.n_input))
                self.res_w = self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                                                      size=(self.n_res, self.n_res))
            elif self.weights_type == 'complex gaussian':
                self.input_w = self.random_state.normal(loc=0., scale=self.input_scale,
                                                        size=(self.n_res, self.n_input)) + \
                    1j * self.random_state.normal(loc=0., scale=self.input_scale, size=(self.n_res, self.n_input))
                self.res_w = self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                                                      size=(self.n_res, self.n_res)) + \
                    1j * self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(self.n_res),
                                                  size=(self.n_res, self.n_res))

    def reset(self):
        """ Resets the reservoir state, for new runs """
        self.state = self.random_state.normal(loc=0., scale=1, size=(self.n_res,))

    def encode(self, mat):
        """ Encodes the input before being fed in the reservoir """
        if self.encoding_method == 'binary':
            return mat > self.encoding_param
        elif self.encoding_method == 'phase':
            return np.exp(1j * mat * self.encoding_param)
        elif self.encoding_method is None:
            return mat

    def activation(self):
        """ Activation function for reservoir iterations """
        if self.activation_fun == 'tanh':
            return lambda x: np.tanh(x)
        elif self.activation_fun == 'phase':
            return lambda x: np.exp(1j * x * self.activation_param)

    def iterate(self, input_data):
        """ Iterates the reservoir feeding input_data, returns all the reservoir states """
        sequence_length, input_size = input_data.shape

        concat_states = np.empty((sequence_length-self.forget, self.n_res))
        act = self.activation()

        for time_step in range(sequence_length):
            if self.random_projection == 'simulation':
                self.state = act(np.dot(self.input_w, input_data[time_step, 0:1]) +
                                 np.dot(self.res_w, self.state))
            if time_step >= self.forget:
                concat_states[time_step-self.forget, :] = self.state
        return concat_states

    def train(self, concat_states, y):
        """ Performs a linear regression """
        if self.train_method == 'explicit':
            output_w, res, rnk, s = lstsq(concat_states, y)
            return output_w

    def output(self, concat_states):
        """ Computes the output given reservoir states and output weights """
        return np.dot(concat_states, self.output_w)

    def fit(self, input_data, y=None):
        """ Iterates the reservoir with training input, fits the output weights """
        self.initialize()
        enc_input_data = self.encode(input_data)
        concat_states = self.iterate(enc_input_data)  # shape (sequence_length, n_res)
        self.output_w = self.train(concat_states, y[self.forget:])
        return self

    def predict(self, input_data):  # , y=None):
        """ Iterates the reservoir with input, predicts using the fit output weights """
        self.reset()
        enc_input_data = self.encode(input_data)
        concat_states = self.iterate(enc_input_data)  # shape (sequence_length, n_res)
        res = self.output(concat_states)
        return res
