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
import data_utils
import scipy.io as sio


class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_res=400, res_scale=1, res_encoding=None, res_enc_dim=1, res_enc_param=None,  # reservoir
                 input_scale=1, input_dim=1, input_encoding=None, input_enc_dim=1, input_enc_param=None,  # input
                 add_bias=True, bias_scale=1,  # bias
                 random_projection='simulation', weights_type='gaussian',  # weights
                 activation_fun='tanh', activation_param=None, leak_rate=1,  # dynamics
                 parallel_runs=None, forget=100,  # iterations
                 future_pred=True, pred_horizon=10, rec_pred_steps=0,  # prediction
                 train_method='ridge', train_param=1e1,  # fit
                 cam_roi=None, cam_sampling_range=None, slm_size=None, matlab_eng=None, # SLM experiment
                 random_state=None, save=0, verbose=1):  # misc
        self.n_res = n_res
        self.res_scale = res_scale
        self.res_encoding = res_encoding
        self.res_enc_dim = res_enc_dim
        self.res_enc_param = res_enc_param
        self.input_scale = input_scale
        self.input_encoding = input_encoding
        self.input_enc_dim = input_enc_dim
        self.input_enc_param = input_enc_param
        self.add_bias = add_bias
        self.bias_scale = bias_scale
        self.random_projection = random_projection
        self.weights_type = weights_type
        self.activation_fun = activation_fun
        self.activation_param = activation_param
        self.leak_rate = leak_rate
        self.parallel_runs = parallel_runs
        self.forget = forget
        self.future_pred = future_pred
        self.pred_horizon = pred_horizon
        self.rec_pred_steps = rec_pred_steps
        self.train_method = train_method
        self.train_param = train_param
        self.random_state = random_state
        self.save = save
        self.verbose = verbose
        self.cam_roi = cam_roi
        self.slm_size = slm_size

        self.input_dim = None
        self.input_w = None
        self.res_w = None
        self.output_w = None
        self.state = None

        self.fit_score = None
        self.init_timer = None
        self.iterate_timer = None
        self.train_timer = None

        if self.random_projection == 'lighton opu':
            from lightonopu.opu import OPU
            from lightonml.random_projections.opu import OPURandomMapping

            self.opu = OPU()
            self.random_mapping = OPURandomMapping(opu=self.opu, n_components=self.n_res)
            # Use "disable_pbar=True" if needed
            self.parallel_runs = 100
        elif self.random_projection == 'meadowlark slm':
            self.cam_sampling_range = None
            self.matlab_eng = matlab_eng
            self.cam_sampling_range = cam_sampling_range
        elif self.random_projection == 'out of core' and self.weights_type == 'complex gaussian':
            self.input_w_re = None
            self.input_w_im = None
            self.res_w_re = None
            self.res_w_im = None

    def fit(self, input_data, y=None):
        """
        Iterates the reservoir with training input and fits the output weights based on the first n time steps of
        input_data in order to predict next time steps with length of pred_length, for each n.
        """
        self.input_dim = input_data.shape[-1]
        start = time.time()
        if self.verbose:
            print('Reservoir Computing algorithm - Training phase:\n')
        self.initialize()
        self.reset_state()
        enc_input_data = self.encode_input(input_data)
        # If reservoir is in prediction mode, generate the output
        if self.future_pred and y is None:
            y = data_utils.roll_and_concat(input_data, roll_num=self.pred_horizon)
        init_end = time.time()
        self.init_timer = init_end - start
        if self.verbose:
            print('Initialization complete. \t\tElapsed time: ' + str(self.init_timer) + ' s')

        concat_states = self.iterate(enc_input_data)
        iterate_end = time.time()
        self.iterate_timer = iterate_end - init_end
        if self.verbose:
            print('Reservoir iterations complete. \t\tElapsed time: ' + str(self.iterate_timer) + ' s')

        true_output = y[:, self.forget:, :]
        self.output_w = self.train(concat_states, true_output)

        pred_output = self.output(concat_states)
        self.fit_score = self.score_metric(pred_output, true_output)

        if self.verbose:
            train_end = time.time()
            self.train_timer = train_end - iterate_end
            print('Training complete. \t\t\tElapsed time: ' + str(self.train_timer) + ' s')
            print('Training score: ' + str(self.fit_score))
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


    def predict_and_score(self, input_data, true_output=None, only_score=False, detailed_score=False):
        # If reservoir is in prediction mode, generate the output
        if self.future_pred and true_output is None:
            true_output = data_utils.roll_and_concat(input_data, roll_num=self.pred_horizon)

        # Use Reservoir to predict the output
        start = time.time()
        if self.verbose:
            print('Reservoir Computing algorithm - Testing phase:\n')
        self.reset_state()
        enc_input_data = self.encode_input(input_data)
        init_end = time.time()
        init_timer = init_end - start
        if self.verbose:
            print('Initialization complete. \t\tElapsed time: ' + str(init_timer) + ' s')

        concat_states = self.iterate(enc_input_data)  # shape (sequence_length, n_res)
        iterate_end = time.time()
        iterate_timer = iterate_end - init_end
        if self.verbose:
            print('Reservoir iterations complete. \t\tElapsed time: ' + str(iterate_timer) + ' s')

        pred_output = self.output(concat_states)

        # Compare with true output
        true_output = true_output[:, self.forget:, :]
        true_output = true_output.reshape(-1, true_output.shape[-1])
        pred_output = pred_output.reshape(-1, pred_output.shape[-1])
        score = self.score_metric(pred_output, true_output)

        test_end = time.time()
        test_timer = test_end - iterate_end
        if self.verbose:
            print('Testing complete. \t\t\tElapsed time: ' + str(test_timer) + ' s')
            print('Testing score: ' + str(score))

        if self.future_pred and detailed_score:
            score = self.detailed_pred_score(pred_output, true_output)
        else:
            score = self.score_metric(pred_output, true_output)

        if only_score:
            return score
        else:
            return pred_output, score


    def score(self, input_data, true_output=None, sample_weight=None):

        return self.predict_and_score(input_data, true_output, only_score=True)


    def initialize(self):
        """ Initializes the reservoir state, the input and reservoir weights """
        self.random_state = check_random_state(self.random_state)
        total_input_dim = self.input_dim * self.input_enc_dim
        total_res_dim = self.n_res * self.res_enc_dim
        if self.random_projection == 'simulation':
            self.bias_vec = self.add_bias*self.random_state.normal(loc=0., scale=self.bias_scale, size=(self.n_res, 1))
            if self.weights_type == 'gaussian':
                self.input_w = self.random_state.normal(loc=0., scale=self.input_scale / np.sqrt(total_input_dim),
                                                        size=(self.n_res, total_input_dim))
                self.res_w = self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(total_res_dim),
                                                      size=(self.n_res, total_res_dim))
            elif self.weights_type == 'complex gaussian':
                self.input_w = 1j * self.random_state.normal(loc=0., scale=self.input_scale / np.sqrt(total_input_dim),
                                                             size=(self.n_res, total_input_dim))
                self.input_w += self.random_state.normal(loc=0., scale=self.input_scale / np.sqrt(total_input_dim),
                                                         size=(self.n_res, total_input_dim))
                self.res_w = 1j * self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(total_res_dim),
                                                           size=(self.n_res, total_res_dim))
                self.res_w += self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(total_res_dim),
                                                       size=(self.n_res, total_res_dim))
        elif self.random_projection == 'out of core':
            self.bias_vec = self.add_bias*self.random_state.normal(loc=0., scale=self.bias_scale, size=(self.n_res, 1))
            n_batch = 2
            step = int(self.n_res / n_batch)
            if self.weights_type == 'gaussian':
                self.input_w = np.memmap(
                    'data/input_w.dat', dtype='float32', mode='w+', shape=(self.n_res, total_input_dim))
                self.res_w = np.memmap('data/res_w.dat', dtype='float32', mode='w+', shape=(self.n_res, total_res_dim))

                for i_batch in range(n_batch):
                    self.input_w[i_batch * step:(i_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(total_input_dim),
                                                 size=(step, total_input_dim))
                    for j_batch in range(n_batch):
                        self.res_w[i_batch * step:(i_batch+1) * step, j_batch * step:(j_batch+1) * step] = \
                            self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(total_res_dim),
                                                     size=(step, step*self.res_enc_dim))
            elif self.weights_type == 'complex gaussian':
                self.input_w_re = np.memmap('data/input_w_re.dat', dtype='float32', mode='w+',
                                            shape=(self.n_res, total_input_dim))
                self.input_w_im = np.memmap('data/input_w_im.dat', dtype='float32', mode='w+',
                                            shape=(self.n_res, total_input_dim))
                self.res_w_re = np.memmap('data/res_w_re.dat', dtype='float32', mode='w+',
                                          shape=(self.n_res, total_res_dim))
                self.res_w_im = np.memmap('data/res_w_im.dat', dtype='float32', mode='w+',
                                          shape=(self.n_res, total_res_dim))

                for i_batch in range(n_batch):
                    self.input_w_re[i_batch * step:(i_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(total_input_dim),
                                                 size=(step, total_input_dim))
                    self.input_w_im[i_batch * step:(i_batch+1) * step] = \
                        self.random_state.normal(loc=0., scale=self.input_scale/np.sqrt(total_input_dim),
                                                 size=(step, total_input_dim))
                    for j_batch in range(n_batch):
                        self.res_w_re[i_batch * step:(i_batch+1) * step, j_batch * step:(j_batch+1) * step] = \
                            self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(total_res_dim),
                                                     size=(step, step*self.res_enc_dim))
                        self.res_w_im[i_batch * step:(i_batch+1) * step, j_batch * step:(j_batch+1) * step] = \
                            self.random_state.normal(loc=0., scale=self.res_scale / np.sqrt(total_res_dim),
                                                     size=(step, step*self.res_enc_dim))

    def reset_state(self):
        """ Resets the reservoir state, for new runs """
        # To-do: add different statistics
        if self.parallel_runs is None:
            self.state = self.random_state.normal(loc=0., scale=1, size=self.n_res)
        else:
            self.state = self.random_state.normal(loc=0., scale=1, size=(self.n_res, self.parallel_runs))

    def encode_input(self, mat):
        """ Encodes the input of the reservoir """
        if self.input_encoding == 'threshold':
            if self.input_enc_param is None:
                self.input_enc_param = 25
            return encode.binary_threshold(mat, self.input_enc_param)
        elif self.input_encoding == 'naive binary':
            if self.input_enc_param is None:
                self.input_enc_param = [-0.5, 0.5]
            return encode.naive_binary(mat, binary_dim=self.input_enc_dim,
                                       lower_bound=self.input_enc_param[0], higher_bound=self.input_enc_param[1])
        elif self.input_encoding == 'local binary':
            if self.input_enc_param is None:
                self.input_enc_param = [-0.5, 0.5, 0.5]
            return encode.local_binary(mat, binary_dim=self.input_enc_dim, lower_bound=self.input_enc_param[0],
                                       higher_bound=self.input_enc_param[1], step=self.input_enc_param[2])
        elif self.input_encoding == 'phase':
            if self.input_enc_param is None:
                self.input_enc_param = np.pi
            return encode.phase_encoding(mat, scaling_factor=self.input_enc_param, n_levels=255)
        elif self.input_encoding == 'meadowlark slm':
            if self.input_enc_param is None:
                self.input_enc_param = int(256/2)
            return encode.slm_encoding(mat, scaling_factor=self.input_enc_param, n_levels=int(256/2))
        elif self.input_encoding == 'fixed binary':
            if self.input_enc_param is None:
                self.input_enc_param = [0, 1]
            return encode.fixed_binary(mat, binary_dim=self.input_enc_dim, lower_bound=self.input_enc_param[0],
                                       higher_bound=self.input_enc_param[1])
        elif self.input_encoding == 'large bin binary':
            if self.input_enc_param is None:
                self.input_enc_param = [-0.5, 0.5, 0.25, False]
            return encode.large_bin_binary(mat, binary_dim=self.input_enc_dim, lower_bound=self.input_enc_param[0],
                                           higher_bound=self.input_enc_param[1], gamma=self.input_enc_param[2],
                                           balanced=self.input_enc_param[3])
        elif self.input_encoding is None:
            return mat

    def encode_res(self, mat):
        """ Encodes the input of the reservoir """
        if self.res_encoding == 'threshold':
            if self.res_enc_param is None:
                self.res_enc_param = 25
            return encode.binary_threshold(mat, self.res_enc_param)
        elif self.res_encoding == 'phase':
            if self.res_enc_param is None:
                self.res_enc_param = np.pi
            return encode.phase_encoding(mat, scaling_factor=self.res_enc_param)
        elif self.res_encoding == 'meadowlark slm':
            if self.res_enc_param is None:
                self.res_enc_param = int(256/2)
            return encode.slm_encoding(mat, scaling_factor=self.res_enc_param, n_levels=int(256/2))
        elif self.res_encoding == 'naive binary':
            if self.res_enc_param is None:
                self.res_enc_param = [-0.5, 0.5]
            return encode.naive_binary(mat.T, binary_dim=self.res_enc_dim,
                                       lower_bound=self.res_enc_param[0], higher_bound=self.res_enc_param[1]).T
        elif self.res_encoding == 'local binary':
            if self.res_enc_param is None:
                self.res_enc_param = [-0.5, 0.5, 0.5]
            return encode.local_binary(mat.T, binary_dim=self.res_enc_dim, lower_bound=self.res_enc_param[0],
                                       higher_bound=self.res_enc_param[1], step=self.res_enc_param[2]).T
        elif self.res_encoding == 'fixed binary':
            if self.input_enc_param is None:
                self.input_enc_param = [0, 1]
            return encode.fixed_binary(mat.T, binary_dim=self.input_enc_dim, lower_bound=self.input_enc_param[0],
                                       higher_bound=self.input_enc_param[1]).T
        elif self.res_encoding == 'large bin binary':
            if self.res_enc_param is None:
                self.res_enc_param = [0, 1, 0.25, False]
            return encode.large_bin_binary(mat.T, binary_dim=self.res_enc_dim, lower_bound=self.res_enc_param[0],
                                           higher_bound=self.res_enc_param[1], gamma=self.res_enc_param[2],
                                           balanced=self.res_enc_param[3]).T
        elif self.res_encoding is None:
            return mat

    def activation(self):
        """ Activation function for reservoir iterations """
        if self.activation_fun == 'tanh':
            return lambda x: np.tanh(x)
        elif self.activation_fun == 'phase':
            return lambda x: np.exp(1j * x / np.amax(np.abs(x)) * self.activation_param)
        elif self.activation_fun == 'abs phase':
            return lambda x: np.exp(1j * np.abs(x) / np.amax(np.abs(x)) * self.activation_param)
        elif self.activation_fun == 'fixed phase':
            return lambda x: np.exp(1j * x * self.activation_param)
        elif self.activation_fun == 'phase 8bit':
            def fun(x):
                x = np.array(np.abs(x) / np.amax(np.abs(x)) * 255, dtype='int') / 255
                return np.exp(1j * x * self.activation_param)
            return fun
        elif self.activation_fun == 'binary threshold':
            return lambda x: np.abs(x) > np.median(np.abs(x))  # to activate the half of the neurons
        elif self.activation_fun == 'abs':
            return lambda x: np.abs(x)
        elif self.activation_fun == 'intensity':
            if self.activation_param is None:
                self.activation_param = 1
            def fun(x):
                x = np.array(np.abs(x) ** 2)
                return x * (x < self.activation_param) + self.activation_param * (x > self.activation_param)
            return fun  # lambda x: np.abs(x) ** 2
        else:  # in this last case, we allow external definition of the activation function
            return self.activation_fun

    def iterate(self, input_data):
        """ Iterates the reservoir and return all the successive reservoir states """
        n_sequence, sequence_length, input_dim = input_data.shape

        input_iscomplex = True if any(np.iscomplex(input_data.flatten())) else False
        state_iscomplex = True if self.activation_fun == 'phase' or self.activation_fun == 'phase 8bit' else False
        n_parallel = self.parallel_runs if self.parallel_runs is not None else 1
        concat_states = np.zeros((n_sequence, sequence_length-self.forget,
                                  self.n_res+input_dim), dtype=np.float64)
        act = self.activation()

        # Initialize hardware if we use the optical setup
        if self.random_projection == 'lighton opu':
            self.opu.open()
        elif self.random_projection == 'meadowlark slm':
            if self.matlab_eng is None:
                import matlab.engine
                self.matlab_eng = matlab.engine.start_matlab()
                if self.cam_roi is None:
                    self.cam_roi = [350, 350]
                self.matlab_eng.workspace['cam_roi'] = matlab.double(self.cam_roi)
                if self.slm_size is None:
                    self.slm_size = [512, 512]
                self.matlab_eng.workspace['slm_size'] = matlab.double(self.slm_size)
                self.cam_sampling_range = np.linspace(0, (self.cam_roi[0]-1)*(self.cam_roi[1]-1)-1, self.n_res, dtype='uint32')
                self.matlab_eng.cd(r'D:\Users\Mickael-manip\Desktop\JonMush', nargout=0)
                self.matlab_eng.open_slm(nargout=0)
                self.matlab_eng.open_maitai(nargout=0)
                self.matlab_eng.open_camera(nargout=0)

        for i_sequence in range(int(n_sequence / n_parallel)):
            if self.parallel_runs is not None:
                idx_sequence = np.arange(i_sequence * self.parallel_runs, (i_sequence + 1) * self.parallel_runs)
            else:
                idx_sequence = i_sequence
            if self.verbose:
                time_iterable = tqdm(range(sequence_length), file=sys.stdout)
            else:
                time_iterable = range(sequence_length)
            for time_step in time_iterable:
                if self.random_projection == 'simulation':
                    previous_state = self.state
                    self.state = self.encode_res(self.state)
                    self.state = self.leak_rate * act(np.dot(self.input_w, input_data[idx_sequence, time_step, :].T) +
                                    np.dot(self.res_w, self.state) + self.bias_vec) + \
                                 (1 - self.leak_rate) * previous_state
                elif self.random_projection == 'hyperdimensional':
                    # Remove the reservoir weights
                    previous_state = self.state
                    self.state = self.encode_res(self.state)
                    self.state = self.leak_rate * act(np.dot(self.input_w, input_data[idx_sequence, time_step, :].T) +
                                    + self.bias_vec)+ \
                                 (1 - self.leak_rate) * np.roll(previous_state, 1, axis=0);
                elif self.random_projection == 'lighton opu':
                    self.state = self.encode_res(self.state)
                    self.state = self.random_mapping.fit_transform(np.concatenate(
                        (self.state, input_data[idx_sequence, time_step, :])))
                elif self.random_projection == 'meadowlark slm':
                    self.state = self.encode_res(self.state)
                    slm_imgs = self.generate_slm_imgs(input_data[idx_sequence, time_step, :], self.state)
                    if n_parallel==1:
                        adict = {}
                        adict['phase_vec'] = np.array(slm_imgs[0,:], dtype='uint8') # since SLM is 8bit
                        sio.savemat('phase_vec.mat', adict)
                        self.matlab_eng.get_speckle(nargout=0)
                        cam_data_matlab = self.matlab_eng.workspace['data']
                        self.state = (1-self.leak_rate)*np.ravel(np.array(cam_data_matlab._data).reshape(
                            cam_data_matlab.size[::-1]).T)[self.cam_sampling_range] + self.leak_rate*self.state
                    else:
                        for i_img in range(n_parallel):
                            raise ValueError("n_parallel is greater than one")
                            
#                             adict = {}
#                             adict['phase_vec'] = np.array(slm_imgs[i_img,:], dtype='uint8') # since SLM is 8bit
#                             sio.savemat('phase_vec.mat', adict)
#                             self.matlab_eng.get_speckle(nargout=0)
#                             cam_data_matlab = self.matlab_eng.workspace['data']
#                             self.state = np.ravel(np.array(cam_data_matlab._data).reshape(
#                                 cam_data_matlab.size[::-1]).T)[self.cam_sampling_range]
                if time_step >= self.forget:
                    state = np.angle(self.state, deg=False) if state_iscomplex else self.state
                    inputdata = np.angle(input_data[idx_sequence, time_step, :], deg=False) \
                        if input_iscomplex else input_data[idx_sequence, time_step, :]
                    concat_states[idx_sequence, time_step - self.forget, :] = np.concatenate((state, inputdata.T)).T
                    self.concat = input_data

        # Release hardware if we use the optical setup
        if self.random_projection == 'lighton opu':
            self.opu.close()
#         elif self.random_projection == 'meadowlark slm':
#             self.matlab_eng.close_all(nargout=0)
#             self.matlab_eng = None

        if self.verbose >= 2:
            res_states = concat_states[:, :, :n_complex*self.n_res]
            min_res = np.amin(res_states)
            max_res = np.amax(res_states)
            mean_res = np.mean(res_states)
            std_res = np.std(res_states)
            print('Statistics of the reservoir states:')
            print('Sample ' + str(res_states[-1, -1, :5]))
            print('Mean value: ' + str(mean_res))
            print('Standard deviation: ' + str(std_res))
            print('Minimal value: ' + str(min_res))
            print('Maximal value: ' + str(max_res))
            if self.verbose >= 3:
                import matplotlib.pyplot as plt
                plt.hist(np.ravel(res_states[-1, -100::10, :self.n_res]), bins='auto')
                plt.title('Distribution of reservoir activations')
                plt.xlabel('Activation value')
                plt.show()
        return concat_states

    def train(self, concat_states, y):
        """ Performs a linear regression """
        concat_states = concat_states.reshape(-1, concat_states.shape[-1])
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
            clf = sklearn.linear_model.Ridge(fit_intercept=False, alpha=self.train_param)
            clf.fit(concat_states, y)
            return clf.coef_.T
        elif self.train_method == 'sgd':
            concat_states = np.real_if_close(concat_states, tol=1e5)
            clf = sklearn.linear_model.SGDRegressor(fit_intercept=False, max_iter=50000,
                                                    tol=1e-5, alpha=self.train_param)
            clf.fit(concat_states, y)
            return clf.coef_.T
        elif self.train_method == 'random regression':
            def random_regression(X, y, order=0):
                n, d = X.shape
                stack = np.zeros((d, order + 1))
                beta = np.zeros(d)

                norm_factor = np.mean(np.sum(np.abs(X)**2, axis=1))
                # code if X^T X is invertible
                cov = X.T @ X / norm_factor
                substraction = cov - np.eye(d)
                product = X.T @ y / norm_factor
                beta = X.T @ y / norm_factor
                for i_order in range(order):
                    product = substraction @ product
                    beta += (-1) ** (i_order + 1) * product
                return beta
            concat_states = np.real_if_close(concat_states, tol=1e5)
            return random_regression(concat_states, y, order=self.train_param)

    def output(self, concat_states):
        """ Computes the output given reservoir states and output weights """
        total_output = np.dot(concat_states, self.output_w)
        return total_output

    def recursive_predict(self, input_data):
        """ Feedback the prediction as input, to predict the future of the input time series """
        start = time.time()
        if self.verbose:
            print('Start of testing...')
        self.reset_state()
        self.forget = 0
        enc_input_data = self.encode_input(input_data)
        encode_end = time.time()
        encode_timer = encode_end - start
        if self.verbose:
            print('Initialization finished. Elapsed time:')
            print(encode_timer)

        concat_states = self.iterate(enc_input_data)  # shape (sequence_length, n_res)
        iterate_end = time.time()
        iterate_timer = iterate_end - encode_end
        if self.verbose:
            print('Iterations finished. Elapsed time:')
            print(iterate_timer)

        # Recursive prediction: use the reservoir prediction as input
        n_sequence, sequence_length, input_dim = input_data.shape
        output = np.zeros((n_sequence, sequence_length+self.pred_horizon*self.rec_pred_steps, input_dim))
        reservoir_output = self.output(concat_states)
        # Put all the next-time-step prediction in output (starting from 1 since 0 is not predicted)
        output[:, 1:sequence_length, :] = reservoir_output[:, :-1, :input_dim]
        output[:, 0, :] = input_data[:, 0, :]  # We put the original input_data for the first point
        # Use the last state to predict the future
        output[:, sequence_length:sequence_length+self.pred_horizon, :] = \
            np.reshape(reservoir_output[:, -1, :], (n_sequence, self.pred_horizon, input_dim))
        for i_rec_step in range(1, self.rec_pred_steps):
            next_input = reservoir_output[:, -1, :].reshape((n_sequence, self.pred_horizon, input_dim))
            enc_next_input = self.encode_input(next_input)
            next_concat_states = self.iterate(enc_next_input)
            reservoir_output = self.output(next_concat_states)
            output[:, sequence_length+i_rec_step*self.pred_horizon:
                   sequence_length+(i_rec_step+1)*self.pred_horizon, :] = \
                np.reshape(reservoir_output[:, -1, :], (n_sequence, self.pred_horizon, input_dim))

        test_end = time.time()
        test_timer = test_end - iterate_end
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(test_timer)
        return output

    def recursive_predict_score(self, input_data):
        """ Feedback the prediction as input, to predict the future of the input time series """
        start = time.time()
        if self.verbose:
            print('Start of testing...')
        trunc_input_data = input_data[:, :-self.pred_horizon*self.rec_pred_steps, :]
        self.reset_state()
        self.forget = 0
        enc_input_data = self.encode_input(trunc_input_data)
        encode_end = time.time()
        encode_timer = encode_end - start
        if self.verbose:
            print('Initialization finished. Elapsed time:')
            print(encode_timer)

        concat_states = self.iterate(enc_input_data)  # shape (sequence_length, n_res)
        iterate_end = time.time()
        iterate_timer = iterate_end - encode_end
        if self.verbose:
            print('Iterations finished. Elapsed time:')
            print(iterate_timer)

        n_sequence, sequence_length, input_dim = trunc_input_data.shape
        if n_sequence != 1:
            raise('The number of sequences should be equal to 1 in recursive prediction mode.')

        # Change to parallel mode
        n_parallel = 100
        previous_parallel = self.parallel_runs
        self.parallel_runs = n_parallel
        # Also retrieve the previous states
        self.state = concat_states[0, -n_parallel:, :self.n_res].T

        # Recursive prediction: use the reservoir prediction as input
        output = np.zeros((n_parallel, sequence_length+self.pred_horizon*self.rec_pred_steps, input_dim))
        reservoir_output = self.output(concat_states)
        # Put all the next-time-step prediction in output (starting from 1 since 0 is not predicted)
        for i_parallel in range(n_parallel):
            output[i_parallel, i_parallel+1:sequence_length, :] = reservoir_output[0, :-i_parallel-1, :input_dim]
            output[i_parallel, :i_parallel+1, :] = input_data[:, :i_parallel+1, :]
            output[i_parallel, sequence_length:sequence_length+self.pred_horizon, :] = \
                np.reshape(reservoir_output[0, -i_parallel-1, :], (1, self.pred_horizon, input_dim))
        for i_rec_step in range(1, self.rec_pred_steps):
            next_input = output[:, sequence_length + i_rec_step*self.pred_horizon - 1, :].reshape(n_parallel, 1, -1)
            enc_next_input = self.encode_input(next_input)
            next_concat_states = self.iterate(enc_next_input)
            reservoir_output = self.output(next_concat_states)
            output[:, sequence_length+i_rec_step*self.pred_horizon:
                   sequence_length+(i_rec_step+1)*self.pred_horizon, :] = \
                np.reshape(reservoir_output[:, -1, :], (n_parallel, self.pred_horizon, input_dim))

        test_end = time.time()
        test_timer = test_end - iterate_end
        if self.verbose:
            print('Testing finished. Elapsed time:')
            print(test_timer)

        score_vec = np.zeros((self.pred_horizon * self.rec_pred_steps,))
        error_vec = np.zeros((self.pred_horizon * self.rec_pred_steps,))
        for i_parallel in range(n_parallel):
            vec1 = output[i_parallel, sequence_length:, :]
            vec2 = input_data[0, sequence_length-i_parallel:
                                 sequence_length-i_parallel+self.pred_horizon*self.rec_pred_steps, :]
            error_vec += np.squeeze(np.abs(vec1 - vec2) ** 2)
        error_vec /= n_parallel
        score_vec = 1 - error_vec / np.mean(np.abs(input_data - np.mean(input_data)) ** 2)
        # TODO: test the normalization

        self.parallel_runs = previous_parallel
        return error_vec, output

    @staticmethod
    def score_metric(pred_output, output):
        return 1 - np.sum(np.abs(pred_output-output)**2) / np.sum(np.abs(output-np.mean(output))**2)

    def detailed_pred_score(self, pred_output, output):
        if self.future_pred is False:
            print('The "detailed_pred_score" function should only be called in prediction mode.')
            return -1
        step = int(pred_output.shape[-1] / self.pred_horizon)
        score_vec = np.zeros((self.pred_horizon, ))

        for i_horizon in range(self.pred_horizon):
            score_vec[i_horizon] = 1 - np.sum(np.abs(pred_output[:, i_horizon*step:(i_horizon+1)*step] -
                                                     output[:, i_horizon*step:(i_horizon+1)*step])**2) /\
                                   np.sum(np.abs(output[:, i_horizon*step:(i_horizon+1)*step] -
                                                 np.mean(output[:, i_horizon*step:(i_horizon+1)*step]))**2)

        return score_vec



    def generate_slm_imgs(self, input_data, reservoir):

        if len(input_data.shape)==1:
            input_data = input_data.reshape((1,-1))

        if len(reservoir.shape)==1:
            reservoir = reservoir.reshape((-1,1))

        # We first fix the size of the reservoir
        res_repeat = np.round(self.res_scale**2)
        res_size = res_repeat * self.n_res

        # We find how many times to repeat the input
        n_sequence, input_dim = input_data.shape
        input_repeat = np.round(self.n_res * self.input_scale**2 / input_dim)
        input_size = int(input_repeat * input_dim)

        # We find the bias
        bias_repeat = np.round(self.n_res * self.bias_scale**2)
        bias_size = bias_repeat

        # We put everything in a new vector except the bias since the matlab will automatically add it before to send it to SLM
        total_size = np.int(res_size + input_size)
        factor = int(self.slm_size[0] * self.slm_size[1] / (total_size+bias_size))
#         print(input_data.shape, reservoir.shape, res_size, input_size, factor)
        if not factor:
            raise ValueError(["SLM used pixels are not sufficient. Required at least ", str(total_size+bias_size)])
        else:
            slm_imgs = np.zeros((n_sequence, total_size*factor))
            slm_imgs[:, :res_size*factor] = np.repeat(reservoir.T, res_repeat * factor, axis=1)
            slm_imgs[:, res_size*factor:res_size*factor+input_size*factor] = np.repeat(input_data, input_repeat * factor, axis=1)
            return slm_imgs
            
        