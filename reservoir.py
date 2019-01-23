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
    - encoding_param: TBD
    - activation_fun: activation function for every unit of the reservoir
    - activation_param: TBD
    - forget: number of initial states we discard to get rid of transient states
    - train_method: algorithm for the output regression
    - train_param: TBD
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

# from lightonml.random_projections.opu import OPURandomMapping
# from lightonopu.opu import OPU


class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=1, n_res=100, input_scale=1, res_scale=1,
                 random_projection='simulation', weights_type='gaussian', opu_transform=None,
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
        self.opu_transform = opu_transform
        self.save = save
        self.verbose = verbose

        self.state = None
        self.input_w = None
        self.res_w = None
        self.output_w = None

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
        elif self.random_projection == 'out of core':
            n_batch = 2
            step = int(self.n_res / n_batch)
            if self.weights_type == 'gaussian':
                self.input_w = np.memmap('data/input_w.dat', dtype='float32', mode='w+', shape=(self.n_res, self.input_dim))
                self.res_w = np.memmap('data/res_w.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))

                for i_batch in range(n_batch):
                    self.input_w[i_batch * step : (i_batch+1) * step] = \
                    self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim), 
                        size=(step, self.input_dim))
                    for j_batch in range(n_batch):
                        self.res_w[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim), 
                            size=(step, step))
            elif self.weights_type == 'complex gaussian':
                self.input_w_re = np.memmap('data/input_w_re.dat', dtype='float32', mode='w+', shape=(self.n_res, self.input_dim))
                self.input_w_im = np.memmap('data/input_w_im.dat', dtype='float32', mode='w+', shape=(self.n_res, self.input_dim))
                self.res_w_re = np.memmap('data/res_w_re.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))
                self.res_w_im = np.memmap('data/res_w_im.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))

                for i_batch in range(n_batch):
                    self.input_w_re[i_batch * step : (i_batch+1) * step] = \
                    self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim), 
                        size=(step, self.input_dim))
                    self.input_w_im[i_batch * step : (i_batch+1) * step] = \
                    self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim), 
                        size=(step, self.input_dim))
                    for j_batch in range(n_batch):
                        self.res_w_re[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim), 
                            size=(step, step))
                        self.res_w_im[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.input_dim), 
                            size=(step, step))
            

    def reset(self):
        """ Resets the reservoir state, for new runs """
        self.state = self.random_state.normal(loc=0., scale=1, size=(self.n_res))

    def encode(self, mat):
        """ Encodes the input before being fed in the reservoir """
        if self.encoding_method == 'threshold':
            return mat > self.encoding_param
        elif self.encoding_method == 'phase':
            return np.exp(1j * mat * self.encoding_param)
        elif self.encoding_method == 'naivebinary':
            n_sequence, sequence_length, data_dim = mat.shape

            mini = -self.encoding_param
            maxi = self.encoding_param
            step = (maxi - mini) / np.ceil(self.input_dim / data_dim)
            
            enc_input_data = np.zeros((n_sequence, sequence_length, self.input_dim))
            for i_input in range(self.input_dim):
                i_data = np.mod(i_input, data_dim)
                enc_input_data[:, :, i_input] = mat[:, :, i_data] > mini + np.ceil(i_input / data_dim) * step
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
        n_sequence, sequence_length, input_dim = input_data.shape

        concat_states = np.empty((n_sequence, sequence_length-self.forget, self.n_res+input_dim), dtype='cfloat')
        act = self.activation()

        for i_sequence in range(n_sequence):
            self.reset()
            for time_step in tqdm(range(sequence_length), file=sys.stdout):
                if self.random_projection == 'simulation':
                    self.state = act(np.dot(self.input_w, input_data[i_sequence, time_step, :]) +
                                     np.dot(self.res_w, self.state))
                elif self.random_projection == 'out of core':
                    if self.weights_type == 'gaussian':
                        self.state = act(np.dot(self.input_w, input_data[i_sequence, time_step, :]) + 
                            np.dot(self.res_w, self.state))
                    elif self.weights_type == 'complex gaussian':
                        self.state = act(np.dot(self.input_w_re, input_data[i_sequence, time_step, :]) + 
                            1j * np.dot(self.input_w_im, input_data[i_sequence, time_step, :]) +
                            np.dot(self.res_w_re, self.state) +
                            1j * np.dot(self.res_w_im, self.state))
                if time_step >= self.forget:
                    concat_states[i_sequence, time_step-self.forget, :] = np.concatenate((self.state, input_data[i_sequence, time_step, :].T))
        print(concat_states.shape)
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
        elif self.train_method == 'explicitsklearn':
            clf = sklearn.linear_model.LinearRegression(fit_intercept=False)
            clf.fit(concat_states, y)
            output_w = clf.coef_.T
        elif self.train_method == 'ridge':
            concat_states = np.real_if_close(concat_states, tol=1e5)
            clf = sklearn.linear_model.Ridge(fit_intercept=False, alpha=self.train_param)
            clf.fit(concat_states, y)
            output_w = clf.coef_.T
        elif self.train_method == 'sgd':
            clf = sklearn.linear_model.SGDRegressor(fit_intercept=False, max_iter=50000, tol=1e-5, alpha=self.train_param)
            clf.fit(concat_states, y)
            output_w = clf.coef_.T

        return output_w

    def output(self, concat_states):
        """ Computes the output given reservoir states and output weights """
        concat_states = concat_states.reshape(-1, concat_states.shape[-1])
        total_output = np.dot(concat_states, self.output_w)
        return total_output
        # return np.reshape(total_output, (n_sequence, sequence_length))

    def score_metric(self, pred_output, output):
        return 1 - np.sum(abs(pred_output-output)**2) / np.sum(abs(output-np.mean(output))**2)

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

        concat_states = self.iterate(enc_input_data) # concat_states.shape is (sequence_length, n_res + self.input_dim)
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

    def score(self, input_data, true_output):
        pred_output = np.real_if_close(self.predict(input_data), tol=1e5)
        true_output = true_output.reshape(-1, true_output.shape[-1])
        score = self.score_metric(pred_output, true_output)
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(self.train_timer)
            print('Testing score:')
            print(score)
        return pred_output, score
