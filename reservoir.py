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
    - input_dim: dimension of the input
    - n_res: size of the reservoir
    - input_scale: scale of the input weights
    - res_scale: scale of the res_scale (after normalization by 1/np.sqrt(n_res))
    - random_projection: random_projection type ('simulation', 'experiment')
    - weights_type: distribution of input and reservoir weights
    - encoding_method: encoding technique applied on the input fed to the reservoir
    - encoding_param: parameters for the encoding technique
    - activation_fun: activation function for every unit of the reservoir
    - activation_param: parameters for the activation function
    - forget: number of initial states we discard to get rid of transient states
    - train_method: algorithm for the output regression
    - train_param: parameters for the regression function
    - random_state: random seed for reproducible results

Internal attributes:
    - state: current state of the reservoir, np array (n_res,)
    - input_w: input weights, np array (n_res, input_dim)
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
import time
from tqdm import tqdm
import sys
import encode

# from lightonml.random_projections.opu import OPURandomMapping
# from lightonopu.opu import OPU


class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=1, n_res=100, input_scale=1, res_scale=1,
                 random_projection='simulation', weights_type='gaussian',
                 encoding_method=None, encoding_param=None,
                 activation_fun='tanh', activation_param=None, forget=100,
                 train_method='explicit', train_param=None,
                 random_state=None, save=0, verbose=1):
        self.input_dim = input_dim
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
        self.save = save
        self.verbose = verbose

        self.state = None
        self.input_w = None
        self.res_w = None
        self.output_w = None
        self.fit_score = None
        self.encode_timer = None
        self.iterate_timer = None
        self.train_timer = None

        if self.random_projection == 'opu':
            from lightonopu.opu import OPU
            from lightonml.random_projections.opu import OPURandomMapping

            self.opu = OPU()
            self.opu_n_components = self.n_res  # number of random projections
            self.random_mapping = OPURandomMapping(opu=self.opu, n_components=self.opu_n_components)
            # Use "disable_pbar=True" if needed

    def initialize(self):
        """ Initializes the reservoir state, the input and reservoir weights """
        self.random_state = check_random_state(self.random_state)

        if self.random_projection == 'simulation':
            if self.weights_type == 'gaussian':
                self.input_w = self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim),
                                                        size=(self.n_res, self.input_dim))
                self.res_w = self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                                                      size=(self.n_res, self.n_res))
            elif self.weights_type == 'complex gaussian': 
                self.input_w = 1j * self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim),
                                                             size=(self.n_res, self.input_dim))
                self.input_w += self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim),
                                                         size=(self.n_res, self.input_dim))
                self.res_w = 1j * self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                                                           size=(self.n_res, self.n_res))
                self.res_w += self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(self.n_res),
                                                       size=(self.n_res, self.n_res))

    def reset(self):
        """ Resets the reservoir state, for new runs """
        self.state = self.random_state.normal(loc=0., scale=1, size=self.n_res)

    def encode(self, mat):
        """ Encodes the input before being fed in the reservoir """
        if self.encoding_method == 'threshold':
            return encode.binary_threshold(mat, self.encoding_param)
        elif self.encoding_method == 'phase':
            return encode.phase_encoding(mat, scaling_factor=np.pi, n_levels=None)
        elif self.encoding_method == 'naive_binary':
            n_sequence, sequence_length, data_dim = mat.shape
            return encode.naive_binary(mat, binary_dim=self.input_dim / data_dim)
        elif self.encoding_method == 'local_binary':
            n_sequence, sequence_length, data_dim = mat.shape
            return encode.local_binary(mat, binary_dim=self.input_dim / data_dim)
        elif self.encoding_method is None:
            return mat

    def activation(self):
        """ Activation function for reservoir iterations """
        if self.activation_fun == 'tanh':
            return lambda x: np.tanh(x)
        elif self.activation_fun == 'phase':
            return lambda x: np.exp(1j * np.abs(x) / np.amax(np.abs(x)) * 2 * np.pi)
        elif self.activation_fun == 'phase_8bit':
            def fun(x):
                x = np.array(np.abs(x) / np.amax(np.abs(x)) * 255, dtype='int') / 255
                return np.exp(1j * x * 2 * np.pi)
            return fun
        elif self.activation_fun == 'binary':
            return lambda x: np.abs(x) > np.median(np.abs(x))  # to activate the half of the neurons

    def iterate(self, input_data):
        """ Iterates the reservoir feeding input_data, returns all the reservoir states """
        n_sequence, sequence_length, input_dim = input_data.shape

        n = 2 if np.iscomplex(input_data).any() else 1
        concat_states = np.empty((n_sequence, sequence_length-self.forget, n * (self.n_res+input_dim)), dtype='cfloat')
        act = self.activation()

        # Initialize hardware if we use the optical setup
        if self.random_projection == 'opu':
            self.opu.open()
        for i_sequence in range(n_sequence):
            self.reset()
            for time_step in tqdm(range(sequence_length), file=sys.stdout):
                if self.random_projection == 'simulation':
                    self.state = act(np.dot(self.input_w, input_data[i_sequence, time_step, :]) +
                                     np.dot(self.res_w, self.state))
                elif self.random_projection == 'opu':
                    self.state = (self.state > 25)
                    self.state = self.random_mapping.fit_transform(np.concatenate(
                        (self.state, input_data[i_sequence, time_step, :].T)))
                if time_step >= self.forget:
                    if np.iscomplex(input_data).any():
                        concat_states[i_sequence, time_step - self.forget, :] = np.concatenate(
                            (np.real(self.state), np.imag(self.state), np.real(input_data[i_sequence, time_step, :]).T,
                             np.imag(input_data[i_sequence, time_step, :]).T))
                    else:
                        concat_states[i_sequence, time_step-self.forget, :] = \
                            np.concatenate((self.state, input_data[i_sequence, time_step, :].T))
        # Release hardware if we use the optical setup
        if self.random_projection =='opu':
            self.opu.close()
        return concat_states

    def train(self, concat_states, y):
        """ Performs a linear regression """
        concat_states = concat_states.reshape(-1, concat_states.shape[-1])
        if y.shape[-1] == 1:
            y = np.ravel(y)
        else:
            y = y.reshape(-1, y.shape[-1])

        if self.train_method == 'explicit':
            output_w, res, rnk, s = lstsq(concat_states, y)
            return output_w
        elif self.train_method == 'explicit_sklearn':
            concat_states = np.real_if_close(concat_states, tol=1e5)
            clf = sklearn.linear_model.LinearRegression(fit_intercept=False)
            clf.fit(concat_states, y)
            return clf.coef_.T
        elif self.train_method == 'ridge':
            concat_states = np.real_if_close(concat_states, tol=1e5)
            clf = sklearn.linear_model.Ridge(fit_intercept=False, alpha=self.train_param)
            clf.fit(concat_states, y)
            return clf.coef_.T
        elif self.train_method == 'sgd':
            concat_states = np.real_if_close(concat_states, tol=1e5)
            clf = sklearn.linear_model.SGDRegressor(fit_intercept=False, max_iter=50000,
                                                    tol=1e-5, alpha=self.train_param)
            clf.fit(concat_states, y)
            return clf.coef_.T

    def output(self, concat_states):
        """ Computes the output given reservoir states and output weights """
        concat_states = concat_states.reshape(-1, concat_states.shape[-1])
        total_output = np.dot(concat_states, self.output_w)
        return total_output
        # return np.reshape(total_output, (n_sequence, sequence_length))

    @staticmethod
    def score_metric(pred_output, output):
        return 1 - np.sum(np.abs(pred_output-output)**2) / np.sum(np.abs(output-np.mean(output))**2)

    def fit(self, input_data, y=None):
        """
        Iterates the reservoir with training input and fits the output weights based on the first n time steps of
        input_data in order to predict next time steps with length of pred_length, for each n.
        """

        start = time.time()
        if self.verbose:
            print('Start of training...')
        self.initialize()
        enc_input_data = self.encode(input_data)
        encode_end = time.time()
        self.encode_timer = encode_end - start
        if self.verbose:
            print('Initialization finished. Elapsed time:')
            print(self.encode_timer)

        concat_states = self.iterate(enc_input_data)
        # concat_states.shape is (sequence_length, n_res + self.input_dim)
        iterate_end = time.time()
        self.iterate_timer = iterate_end - start
        if self.verbose:
            print('Iterations finished. Elapsed time:')
            print(self.iterate_timer)

        y_ = y[:, self.forget:, :]
        self.output_w = self.train(concat_states, y_)
        train_end = time.time()
        self.train_timer = train_end - start

        pred_output = self.output(concat_states)
        if y_.shape[-1] == 1:
            true_output = np.ravel(y_)
        else:
            true_output = y_.reshape(-1, y_.shape[-1])
        self.fit_score = self.score_metric(pred_output, true_output)

        if self.verbose:
            print('Training finished. Elapsed time:')
            print(self.train_timer)
            print('Training score:')
            print(self.fit_score)
        if self.save:
            with open('out/concat_states.out', 'w') as f:
                print(concat_states, file=f)
            with open('out/train_y.out', 'w') as f:
                print(y, file=f)
            with open('out/weights.out', 'w') as f:
                print(self.output_w, file=f)
            with open('out/train_predict.out', 'w') as f:
                print(pred_output, file=f)
            if self.verbose:
                print('Results saved in memory.')

        return self

    def predict(self, input_data):  # , y=None):
        """ Iterates the reservoir with input, predicts using the fit output weights """
        start = time.time()
        if self.verbose:
            print('Start of testing...')
        self.reset()
        enc_input_data = self.encode(input_data)
        encode_end = time.time()
        encode_timer = encode_end - start
        if self.verbose:
            print('Initialization finished. Elapsed time:')
            print(encode_timer)

        concat_states = self.iterate(enc_input_data)  # shape (sequence_length, n_res)
        iterate_end = time.time()
        iterate_timer = iterate_end - start
        if self.verbose:
            print('Iterations finished. Elapsed time:')
            print(iterate_timer)

        res = self.output(concat_states)
        test_end = time.time()
        test_timer = test_end - start
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(test_timer)
        return res

    def score(self, input_data, true_output, sample_weight=None):
        pred_output = np.real_if_close(self.predict(input_data), tol=1e5)
        true_output = true_output.reshape(-1, true_output.shape[-1])
        score = self.score_metric(pred_output, true_output)
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(self.train_timer)
            print('Testing score:')
            print(score)
        return pred_output, score
