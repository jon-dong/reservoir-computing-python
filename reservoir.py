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
import sklearn.linear_model


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
                self.input_w = self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                    size=(self.n_res, self.n_input))
                self.res_w = self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                    size=(self.n_res, self.n_res))
            elif self.weights_type == 'complex gaussian':
                self.input_w = 1j * self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input),
                    size=(self.n_res, self.n_input))
                self.input_w += self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                    size=(self.n_res, self.n_input))
                self.res_w = 1j * self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                    size=(self.n_res, self.n_res))
                self.res_w += self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(self.n_res),
                    size=(self.n_res, self.n_res))
        elif self.random_projection == 'out of core':
            n_batch = 2
            step = int(self.n_res / n_batch)
            if self.weights_type == 'gaussian':
                self.input_w = np.memmap('data/input_w.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_input))
                self.res_w = np.memmap('data/res_w.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))

                for i_batch in range(n_batch):
                    self.input_w[i_batch * step : (i_batch+1) * step] = self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                        size=(step, self.n_input))
                    for j_batch in range(n_batch):
                        self.res_w[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                            size=(step, step))
            elif self.weights_type == 'complex gaussian':
                self.input_w_re = np.memmap('data/input_w_re.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_input))
                self.input_w_im = np.memmap('data/input_w_im.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_input))
                self.res_w_re = np.memmap('data/res_w_re.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))
                self.res_w_im = np.memmap('data/res_w_im.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))

                for i_batch in range(n_batch):
                    self.input_w_re[i_batch * step : (i_batch+1) * step] = self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                        size=(step, self.n_input))
                    self.input_w_im[i_batch * step : (i_batch+1) * step] = self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                        size=(step, self.n_input))
                    for j_batch in range(n_batch):
                        self.res_w_re[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                            size=(step, step))
                        self.res_w_im[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.n_input), 
                            size=(step, step))

    def reset(self):
        """ Resets the reservoir state, for new runs """
        self.state = self.random_state.normal(loc=0., scale=1, size=(self.n_res,))

    def encode(self, mat):
        """ Encodes the input before being fed in the reservoir """
        if self.encoding_method == 'binary':
            return mat > self.encoding_param
        elif self.encoding_method == 'phase':
            return np.exp(1j * mat * self.encoding_param)
        elif self.encoding_method == 'realbinary':
            sequence_length, input_size = mat.shape

            mini = -0.55  # self.encoding_param[0]
            maxi = 0.55  # self.encoding_param[1]
            step = (maxi - mini) / self.n_input
            
            enc_input_data = np.zeros((sequence_length, self.n_input))
            for i_input in range(self.n_input):
                enc_input_data[:, i_input] = np.ravel(mat > mini + i_input * step)

            return enc_input_data
        elif self.encoding_method is None:
            return mat

    def activation(self):
        """ Activation function for reservoir iterations """
        if self.activation_fun == 'tanh':
            return lambda x: np.tanh(x)
        elif self.activation_fun == 'phase':
            return lambda x: np.exp(1j * abs(x) * self.activation_param)
        elif self.activation_fun == 'binary':
            return lambda x: abs(x) > self.activation_param

    def iterate(self, input_data):
        """ Iterates the reservoir feeding input_data, returns all the reservoir states """
        sequence_length, input_size = input_data.shape

        concat_states = np.empty((sequence_length-self.forget, self.n_res+self.n_input))
        act = self.activation()

        for time_step in range(sequence_length):
            if self.random_projection == 'simulation':
                self.state = act(np.dot(self.input_w, input_data[time_step, :]) +
                                 np.dot(self.res_w, self.state))
            if self.random_projection == 'out of core':
                if self.weights_type == 'gaussian':
                    self.state = act(np.dot(self.input_w, input_data[time_step, :]) + 
                        np.dot(self.res_w, self.state))
                elif self.weights_type == 'complex gaussian':
                    self.state = act(np.dot(self.input_w_re, input_data[time_step, :]) + 
                        1j * np.dot(self.input_w_im, input_data[time_step, :]) +
                        np.dot(self.res_w_re, self.state) +
                        1j * np.dot(self.res_w_im, self.state))
            if time_step >= self.forget:
                concat_states[time_step-self.forget, :] = np.concatenate([self.state, input_data[time_step, :]])
        return concat_states

    def train(self, concat_states, y):
        """ Performs a linear regression """
        if self.train_method == 'explicit':
            output_w, res, rnk, s = lstsq(concat_states, y)
        elif self.train_method == 'explicitsklearn':
            clf = sklearn.linear_model.LinearRegression(fit_intercept=False)
            clf.fit(concat_states, y)
            output_w = clf.coef_.T
        elif self.train_method == 'ridge':
            clf = sklearn.linear_model.Ridge(fit_intercept=False, alpha=5e1)
            clf.fit(concat_states, y)
            output_w = clf.coef_.T
        elif self.train_method == 'sgd':
            clf = sklearn.linear_model.SGDRegressor(fit_intercept=False, max_iter=50000, tol=1e-5, alpha=5e-1)
            clf.fit(concat_states, y)
            output_w = clf.coef_.T
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
