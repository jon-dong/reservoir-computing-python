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


class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self, encoded_spatial_points, n_res=100, input_scale=1, res_scale=1,
                 random_projection='simulation', weights_type='gaussian', opu_transform=None,
                 encoding_method=None, encoding_param=None, activation_fun='tanh', activation_param=None,
                 forget=100, steps_in_total_pred = 100, steps_in_each_pred = 100, train_method='explicit', train_param=None,
                 random_state=None, save=0, verbose=1):
        self.encoded_spatial_points = encoded_spatial_points
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
        self.steps_in_each_pred = steps_in_each_pred # number of steps algorithm predicts, then updates the reservoir accordingly
        self.steps_in_total_pred = steps_in_total_pred

        self.input_w = None
        self.res_w = None # has to be int^2 in experiment
        self.output_w = None
        self.res_states = None
        self.spatial_points = None
        self.eng = None

    def initialize(self):
        """ Initializes the reservoir state, the input and reservoir weights """
        self.random_state = check_random_state(self.random_state)

        if self.random_projection == 'simulation':
            if self.weights_type == 'gaussian':
                self.input_w = self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                    size=(self.n_res, self.encoded_spatial_points))
                self.res_w = self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                    size=(self.n_res, self.n_res))
            elif self.weights_type == 'complex gaussian':
                self.input_w = 1j * self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                    size=(self.n_res, self.encoded_spatial_points))
                self.input_w += self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                    size=(self.n_res, self.encoded_spatial_points))
                self.res_w = 1j * self.random_state.normal(loc=0., scale=self.res_scale/np.sqrt(self.n_res),
                    size=(self.n_res, self.n_res))
                self.res_w += self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(self.n_res),
                    size=(self.n_res, self.n_res))
        elif self.random_projection == 'out of core':
            n_batch = 2
            step = int(self.n_res / n_batch)
            if self.weights_type == 'gaussian':
                self.input_w = np.memmap(
                    'data/input_w.dat', dtype='float32', mode='w+', shape=(self.n_res, self.encoded_spatial_points))
                self.res_w = np.memmap('data/res_w.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))

                for i_batch in range(n_batch):
                    self.input_w[i_batch * step : (i_batch+1) * step] = \
                    self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                        size=(step, self.encoded_spatial_points))
                    for j_batch in range(n_batch):
                        self.res_w[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                            size=(step, step))
            elif self.weights_type == 'complex gaussian':
                self.input_w_re = np.memmap(
                    'data/input_w_re.dat', dtype='float32', mode='w+', shape=(self.n_res, self.encoded_spatial_points))
                self.input_w_im = np.memmap(
                    'data/input_w_im.dat', dtype='float32', mode='w+', shape=(self.n_res, self.encoded_spatial_points))
                self.res_w_re = np.memmap('data/res_w_re.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))
                self.res_w_im = np.memmap('data/res_w_im.dat', dtype='float32', mode='w+', shape=(self.n_res, self.n_res))

                for i_batch in range(n_batch):
                    self.input_w_re[i_batch * step : (i_batch+1) * step] = \
                    self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                        size=(step, self.encoded_spatial_points))
                    self.input_w_im[i_batch * step : (i_batch+1) * step] = \
                    self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                        size=(step, self.encoded_spatial_points))
                    for j_batch in range(n_batch):
                        self.res_w_re[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                            size=(step, step))
                        self.res_w_im[i_batch * step : (i_batch+1) * step, j_batch * step : (j_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(self.encoded_spatial_points),
                            size=(step, step))


    def encode(self, mat):
        """
        :param mat: shape is (sequence_length x spatial_points)
        :return: Encodes the input before being fed in the reservoir
        """
        if self.encoding_method == 'threshold':
            return mat > self.encoding_param
        elif self.encoding_method == 'phase':
            # mat = np.array((mat - np.amin(mat))/(np.amax(mat) - np.amin(mat))*255, dtype='int')/255

            sequence_length, spatial_points = mat.shape
            slm_enc = 256
            n = int(self.encoded_spatial_points / spatial_points)
            enc = n * slm_enc - 1
            mat = np.array((mat - np.amin(mat))/(np.amax(mat) - np.amin(mat))*enc, dtype='int')
            encoded_mat = np.zeros((sequence_length, n * spatial_points))
            mat0 = np.mod(mat, slm_enc)
            for i in range(n - 1):
                encoded_mat[:, i * spatial_points:(i + 1) * spatial_points] = np.array(
                    (mat - mat0) /slm_enc, dtype=bool)*slm_enc
                mat0 = mat0 + encoded_mat[:, i * spatial_points:(i + 1) * spatial_points]
            encoded_mat[:, (n - 1)*spatial_points:n*spatial_points] = np.mod(mat, slm_enc)
            encoded_mat = encoded_mat/slm_enc
            # encoded_mat = encoded_mat.reshape(
            #     sequence_length, n, spatial_points).T.reshape(n * spatial_points, sequence_length).T
            return np.exp(1j * encoded_mat * 2*np.pi)
        elif self.encoding_method == 'naivebinary':
            sequence_length, spatial_points = mat.shape

            mini = np.min(mat)
            maxi = np.max(mat)
            n_bins = int(np.ceil(self.encoded_spatial_points / spatial_points))
            step = (maxi - mini) / n_bins
            enc_input_data = np.zeros((sequence_length, self.encoded_spatial_points))
            for i_bin in range(n_bins):
                enc_input_data[:, i_bin * spatial_points:(i_bin + 1) * spatial_points] = mat > mini + i_bin * step
            return enc_input_data
        elif self.encoding_method == 'binarybins':
            sequence_length, spatial_points = mat.shape

            mini = np.min(mat)
            maxi = np.max(mat)
            n_bins = int(np.ceil(self.encoded_spatial_points/spatial_points))
            step = (maxi - mini) / n_bins
            enc_input_data = np.zeros((sequence_length, self.encoded_spatial_points))
            for i_bin in range(n_bins):
                enc_input_data[:, i_bin*spatial_points:(i_bin+1)*spatial_points] = np.prod(
                    [mat>mini+i_bin*step , mat<mini+(i_bin+1)*step], axis=0)
            return enc_input_data
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
                x = np.array(np.abs(x) / np.amax(np.abs(x))*255, dtype='int') / 255
                return np.exp(1j*x*2*np.pi)
            return fun
        elif self.activation_fun == 'binary':
            return lambda x: np.abs(x) > np.median(np.abs(x)) # to activate the half of the neurons

    def iterate(self, input_data, update=False):
        """
        :param input_data: n_sequence X sequence_length X spatial_points shape
        :param update: it is false before the prediction, when the reservoir is just starting to fill and it is true
                       during the prediction in order to update the reservoir state after each timestep prediction
        :return: Iterates the reservoir feeding by input_data, returns all the reservoir states
        """
        act = self.activation()

        if self.random_projection == 'optical_setup':
            if self.eng is None:
                import matlab.engine
                import scipy.io as sio
                self.eng = matlab.engine.start_matlab()
                self.eng.cd(r'D:\Users\Mickael-manip\Desktop\JonMush', nargout=0)
                self.eng.open_all(nargout=0)
                cam_dim = np.array([175-np.sqrt(self.n_res)/2, 175+np.sqrt(self.n_res)/2], dtype='int')
                phase_vec = np.zeros((340 * 320))


        if update:
            if self.random_projection == 'simulation':
                print('Updating the reservoir')
                self.forget = 0
                for step in tqdm(range(self.steps_in_each_pred), file=sys.stdout):
                    self.res_states = act(np.dot(
                        self.input_w,
                        input_data[:, step * self.encoded_spatial_points:(step + 1) * self.encoded_spatial_points].T) +
                                          np.dot(self.res_w, self.res_states.T)).T
        else:
            print('Constructing the reservoir')
            sequence_length, _ = input_data.shape
            self.res_states = np.zeros((sequence_length - self.forget, self.n_res), dtype='cfloat')
            # Resets the reservoir state, for new runs
            state = self.random_state.normal(loc=0., scale=1, size=self.n_res)
            for time_step in tqdm(range(sequence_length), file=sys.stdout):
                if self.random_projection == 'simulation':
                    state = act(np.dot(self.input_w, input_data[time_step, :]) + np.dot(self.res_w, state))
                elif self.random_projection == 'optical_setup':
                    phase_vec[:self.n_res] = state
                    phase_vec[self.n_res:self.n_res+self.encoded_spatial_points] = input_data[time_step, :]
                    adict = {}
                    adict['phase_vec'] = np.array(phase_vec.reshape(340,320), dtype='uint8') # since SLM is 8bit
                    sio.savemat('phase_vec.mat', adict)
                    self.eng.get_speckle(nargout=0)
                    cam_data_matlab = self.eng.workspace['data']
                    state = np.ravel(np.array(cam_data_matlab._data).reshape(
                        cam_data_matlab.size[::-1]).T[cam_dim[0]:cam_dim[1], cam_dim[0]:cam_dim[1]])
                elif self.random_projection == 'out of core':
                    if self.weights_type == 'gaussian':
                        state = act(np.dot(
                            self.input_w, input_data[time_step, :]) + np.dot(
                            self.res_w, state[time_step]))
                    elif self.weights_type == 'complex gaussian':
                        state = act(np.dot(
                            self.input_w_re, input_data[time_step, :]) + 1j * np.dot(
                            self.input_w_im, input_data[time_step, :]) + np.dot(
                            self.res_w_re, state) + 1j * np.dot(self.res_w_im, state))
                if time_step >= self.forget:
                    self.res_states[time_step - self.forget, :] = state

    def train(self, concat_states, y):
        """ Performs a linear regression """

        if self.train_method == 'explicit':
            self.output_w, res, rnk, s = lstsq(concat_states, y)
        elif self.train_method == 'explicitsklearn':
            clf = sklearn.linear_model.LinearRegression(fit_intercept=False)
            clf.fit(concat_states, y)
            self.output_w = clf.coef_.T
        elif self.train_method == 'ridge':
            clf = sklearn.linear_model.Ridge(fit_intercept=False, alpha=self.train_param)
            clf.fit(concat_states, y)
            self.output_w = clf.coef_.T
        elif self.train_method == 'sgd':
            clf = sklearn.linear_model.SGDRegressor(
                fit_intercept=False, max_iter=50000, tol=1e-5, alpha=self.train_param)
            clf.fit(concat_states, y)
            self.output_w = clf.coef_.T

    def output(self, concat_states):
        """ Computes the output given reservoir states and output weights """
        total_output = np.dot(concat_states, self.output_w)
        if len(total_output.shape) == 1:
            total_output = total_output.reshape(-1, 1)
        return total_output

    def score_metric(self, pred_output, output):
        return 1 - np.sum(np.abs(pred_output-output)**2) / np.sum(np.abs(output-np.mean(output))**2)

    def fit(self, input_data, y):
        """
        Iterates the reservoir with training input and fits the output weights based on the first n time steps of
        input_data in order to predict next time steps with length of pred_length, for each n.
        """
        start = time.time()
        if self.verbose:
            print('Start of training...')
        self.initialize()

        n_sequence, sequence_length, spatial_points = input_data.shape

        concat_states = self.featurize(input_data[0, :, :].reshape(sequence_length, spatial_points))
        for n in range(n_sequence-1):
            concat_states = np.concatenate((
                concat_states,
                self.featurize(input_data[n+1, :, :].reshape(sequence_length, spatial_points))))

        y = y[:, self.forget:, :].reshape(-1, y.shape[-1])
        if y.shape[-1] == 1:
            true_output = np.ravel(y)
        else:
            true_output = y

        self.train(concat_states, true_output)
        pred_output = self.output(concat_states)

        train_end = time.time()
        self.train_timer = train_end - start

        self.fit_score = self.score_metric(pred_output, y)

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

        if self.random_projection == 'optical_setup':
            self.eng.close_all(nargout=0)
            self.eng = None

        return self

    def predict(self, input_data):
        """
        :param input_data: size of (sequence_length x spatial_points)
        :return: Iterates the reservoir with input, predicts fixed number of timeesteps given by each_prediction_step
                then updates the reservoir by the prediction, predicts the next fixed number of timesteps and repeats.
        """

        sequence_length, self.spatial_points = input_data.shape
        spatial_points_in_each_pred = self.spatial_points * self.steps_in_each_pred
        spatial_points_in_total_pred = self.spatial_points * self.steps_in_total_pred

        start = time.time()
        if self.verbose:
            print('Start of testing...')

        update = False
        total_pred_output = np.zeros((sequence_length-self.forget, spatial_points_in_total_pred))
        steps = int(self.steps_in_total_pred/self.steps_in_each_pred)
        for pred_step in range(steps):
            concat_states = self.featurize(input_data, update=update)
            input_data = self.output(concat_states)
            total_pred_output[
            :, pred_step*spatial_points_in_each_pred:(pred_step+1)*spatial_points_in_each_pred] = input_data
            update = True

        test_end = time.time()
        test_timer = test_end - start
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(test_timer)

        return total_pred_output

    def score(self, input_data, true_output):
        """
        :param input_data: size is (sequence_length x spatial_points)
        :param true_output: size is (sequence_length x spatial_points)
        :return:
        """
        input_data = input_data.reshape(-1, input_data.shape[-1])
        true_output = true_output.reshape(-1, true_output.shape[-1])
        pred_output = self.predict(input_data)
        score = self.score_metric(pred_output, true_output)
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(self.train_timer)
            print('Testing score:')
            print(score)

        if self.random_projection == 'optical_setup':
            self.eng.close_all(nargout=0)
            self.eng = None

        return pred_output, score

    def featurize(self, input_data, update=False):
        start = time.time()
        if update:
            enc_input_data = np.zeros((input_data.shape[0], self.steps_in_each_pred*self.encoded_spatial_points), dtype='cfloat')
            for step in range(self.steps_in_each_pred):
                enc_input_data[:, step*self.encoded_spatial_points:
                                  (step+1)*self.encoded_spatial_points] = self.encode(
                    input_data[:, step*self.spatial_points:(step+1)*self.spatial_points])
        else:
            enc_input_data = self.encode(input_data)
        encode_end = time.time()
        self.encode_timer = encode_end - start
        if self.verbose:
            print('Initialization finished. Elapsed time:')
            print(self.encode_timer)

        self.iterate(enc_input_data, update=update) # calculates or updates self.res_states
        if np.iscomplex(enc_input_data).any():
         # (-self.encoded_spatial_points) ensures that we only concatenate last predicted time serie
            concat_states = np.concatenate((
                np.real(self.res_states), np.imag(self.res_states),
                np.real(enc_input_data[self.forget:, -self.encoded_spatial_points:]),
                np.imag(enc_input_data[self.forget:, -self.encoded_spatial_points:])), axis=1)
        else:
            concat_states = np.concatenate((self.res_states, enc_input_data[self.forget:, -self.encoded_spatial_points:]), axis=1)
        iterate_end = time.time()
        self.iterate_timer = iterate_end - start
        if self.verbose:
            print('Iterations finished. Elapsed time:')
            print(self.iterate_timer)

        return np.real_if_close(concat_states, tol=1e5)
