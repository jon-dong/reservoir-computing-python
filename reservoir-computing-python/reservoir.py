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
from sklearn import preprocessing
import time
from tqdm import tnrange, tqdm_notebook
import sys
import encode
import data_utils
import scipy.io as sio



class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_res=400, parallel_res=1, res_scale=1, res_encoding=None, res_enc_dim=1, res_enc_param=None,  # reservoir
                 input_scale=1, input_dim=None, input_encoding=None, input_enc_dim=1, input_enc_param=None, input_shape=None,  # input
                 input_standardize = False, res_standardize = False, output_standardize = False, # data standardization
                 scale_input_MinMax = False, scale_res_MinMax = False, scale_output_MinMax = False, # data standardization
                 add_bias=True, bias_scale=1,  # bias
                 random_projection='simulation', weights_type='gaussian',  # weights
                 activation_fun='tanh', activation_param=None, leak_rate=1,  # dynamics
                 parallel_runs=None, forget=100,  # iterations
                 future_pred=True, pred_horizon=10, rec_pred_steps=0,  # prediction
                 train_method='ridge', train_param=1e1,  # fit
                 raw_input_feature = False, enc_input_feature = True, # concatenated states properties
                 cam_roi=None, cam_sampling_range=None, slm_size=None, matlab_eng=None, # SLM experiment
                 random_state=None, save=0, verbose=1,  # misc
                 N_0=1, N_1=1, time_change=None, change_type='tanh',  # dynamic activation function options
                 gridsearch=False    # see if we're doing gridsearch
                 ):
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
        self.parallel_runs = parallel_runs if parallel_runs is not None else 1
        self.parallel_runs__ = self.parallel_runs
        self.parallel_res = parallel_res
        self.forget = forget
        self.forget__ = forget
        self.future_pred = future_pred
        self.pred_horizon = pred_horizon
        self.rec_pred_steps = rec_pred_steps
        self.train_method = train_method
        self.input_standardize = input_standardize 
        self.res_standardize = res_standardize 
        self.output_standardize = output_standardize 
        self.scale_input_MinMax = scale_input_MinMax 
        self.scale_res_MinMax = scale_res_MinMax 
        self.scale_output_MinMax = scale_output_MinMax
        self.gridsearch = gridsearch

        self.train_param = train_param
        self.random_state = random_state
        self.raw_input_feature = raw_input_feature
        self.enc_input_feature = enc_input_feature
        self.save = save
        self.verbose = verbose
        self.cam_roi = cam_roi
        self.slm_size = slm_size


        self.N_0 = N_0
        self.N_1 = N_1
        self.time_change = time_change
        self.change_type = change_type
        self.input_shape = input_shape

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

            self.opu = OPU(500, 100)
            self.random_mapping = OPURandomMapping(opu=self.opu, n_components=self.n_res, disable_pbar=True)
            # Use "disable_pbar=True" if needed
        elif self.random_projection == 'meadowlark slm':
            self.cam_sampling_range = None
            self.matlab_eng = matlab_eng
            self.cam_sampling_range = cam_sampling_range
        elif self.random_projection == 'out of core' and self.weights_type == 'complex gaussian':
            self.input_w_re = None
            self.input_w_im = None
            self.res_w_re = None
            self.res_w_im = None

    def fit(self, input_data, true_output=None):
        if self.gridsearch:
            input_data = input_data.reshape((input_data.shape[1], input_data.shape[0], input_data.shape[2]))
        """
        Iterates the reservoir with training input and fits the output weights based on the first n time steps of
        input_data in order to predict next time steps with length of pred_length, for each n.
        """
        if self.input_standardize:
            for i in range(input_data.shape[0]):
                preprocessing.scale(input_data[i, :, :], axis=0, copy=False)
        if self.scale_input_MinMax:
            encode.scale(input_data, self.scale_input_MinMax, in_place=True)
        
        if self.input_dim is None:
            self.input_dim = input_data.shape[-1]
        start = time.time()
        if self.verbose:
            print('Reservoir Computing algorithm - Training phase:\n')
        self.initialize()
        self.reset_state()
        # If reservoir is in prediction mode, generate the output
        if self.future_pred and true_output is None:
            true_output = data_utils.roll_and_concat(input_data, roll_num=self.pred_horizon)
        else:
            if self.output_standardize:
                for i in range(true_output.shape[0]):
                    preprocessing.scale(true_output[i, :, :], axis=0, copy=False)
            if self.scale_output_MinMax:
                encode.scale(true_output, self.scale_output_MinMax, in_place=True)
            if true_output.shape[-1] == self.input_dim and self.pred_horizon != 1:
                true_output = data_utils.roll_and_concat(true_output, roll_num=self.pred_horizon)
            
        if self.parallel_res != input_data.shape[0]/true_output.shape[0]:
            raise ValueError("the number of parallel_res or the ratio of training target sets are selected wrong")

        init_end = time.time()
        self.init_timer = init_end - start
        if self.verbose:
            print('Initialization complete. \t\tElapsed time: ' + str(self.init_timer) + ' s')
        
        concat_states = self.iterate(input_data)
        self.fit_res_state = self.state


        iterate_end = time.time()
        self.iterate_timer = iterate_end - init_end
        if self.verbose:
            print('Reservoir iterations complete. \t\tElapsed time: ' + str(self.iterate_timer) + ' s')

        true_output = true_output[:, self.forget:, :]
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
                print(true_output, file=f)
            with open('out/weights.out', 'w') as f:
                print(self.output_w, file=f)
            with open('out/train_predict.out', 'w') as f:
                print(pred_output, file=f)
            if self.verbose:
                print('Results saved in memory.')
        return self

    def score(self, input_data, true_output=None, sample_weight=None):

        return self.predict_and_score(input_data, true_output, only_score=True)

    def predict_and_score(self, input_data, true_output=None, only_score=False, detailed_score=False, parallel=200, 
                          sample_weight=None):
        if self.gridsearch:
            input_data = input_data.reshape((input_data.shape[1], input_data.shape[0], input_data.shape[2]))
        n_sequence, sequence_length, spatial_points = input_data.shape
        self.forget = sequence_length - self.pred_horizon*self.rec_pred_steps - parallel
        # preprocessing of the input data
        if self.input_standardize:
            for i in range(input_data.shape[0]):
                preprocessing.scale(input_data[i, :, :], axis=0, copy=False)
        if self.scale_input_MinMax:
            encode.scale(input_data, self.scale_input_MinMax, in_place=True)

        # If reservoir is in prediction mode, generate the output
        if self.future_pred and true_output is None:
            true_output = data_utils.roll_and_concat(
                input_data[:, -self.pred_horizon*self.rec_pred_steps-parallel:, :], roll_num=self.pred_horizon*self.rec_pred_steps)
            true_output = true_output[:, :parallel, :]##
        else:
            # preprocessing of the output data
            if self.output_standardize:
                for i in range(true_output.shape[0]):
                    preprocessing.scale(true_output[i, :, :], axis=0, copy=False)
            if self.scale_output_MinMax:
                encode.scale(true_output, self.scale_output_MinMax, in_place=True)
            if true_output.shape[-1] == self.input_dim and self.pred_horizon*self.rec_pred_steps != 1:
                true_output = data_utils.roll_and_concat(
                    true_output[:, self.forget:, :], roll_num=self.pred_horizon*self.rec_pred_steps)
                true_output = true_output[:, :-self.pred_horizon * self.rec_pred_steps, :]

        # Use Reservoir to predict the output
        start = time.time()
        if self.verbose:
            print('Reservoir Computing algorithm - Testing phase:\n')
        self.reset_state()
        init_end = time.time()
        init_timer = init_end - start
        if self.verbose:
            print('Initialization complete. \t\tElapsed time: ' + str(init_timer) + ' s')

        true_output = true_output.reshape(-1, true_output.shape[-1])
        pred_output = np.zeros((parallel, spatial_points*self.pred_horizon*self.rec_pred_steps))##
        # print(pred_output.shape)
        # print('self.forget = '+str(self.forget))
        # print('self.output_w.shape = '+str(self.output_w.shape))
        input_data_temp = input_data[:, :-self.pred_horizon*self.rec_pred_steps, :]##
        # print('input_data_temp='+str(input_data_temp.shape))
        for i in tnrange(self.rec_pred_steps, desc='reservoir update'):
            # print('input_data_temp.shape = '+str(input_data_temp.shape))
            concat_states = self.iterate(input_data_temp).reshape(-1, self.n_res+spatial_points)  # refreshing the concat states for each prediction step
            # print('concat_states='+str(concat_states.shape))
            # print('concat_states='+str(concat_states))
            input_data_temp = self.output(concat_states)
            # print('input_data_temp = '+str(input_data_temp[:3,:4]))

            # if i >= parallel:
            #     j = i - parallel
            #     pred_output[:, j*spatial_points:(j+1)*spatial_points] = input_data_temp

            pred_output[:, i*spatial_points:(i+1)*spatial_points] = input_data_temp
            # print(str(i)+' th pred complete')

        # print('pred_output = '+str(pred_output[:3, :4]))
        # print('true_output = '+str(true_output[:3, :4]))
        iterate_end = time.time()
        iterate_timer = iterate_end - init_end
        if self.verbose:
            print('Reservoir iterations complete. \t\tElapsed time: ' + str(iterate_timer) + ' s')
        try:
            score = self.score_metric(pred_output, true_output)
        except MemoryError:
            print("No sufficient memory for score calculation")
            score = None
        test_end = time.time()
        test_timer = test_end - iterate_end
        if self.verbose:
            print('Testing complete. \t\t\tElapsed time: ' + str(test_timer) + ' s')
            print('Testing score: ' + str(score))
            
        self.forget = self.forget__
        if detailed_score:
            rmse, rmse_vec, rmse_vert = self.detailed_pred_score(pred_output, true_output, spatial_points)
            return pred_output, rmse, rmse_vec, rmse_vert
        elif only_score:
            rmse, rmse_vec, rmse_vert = self.detailed_pred_score(pred_output, true_output, spatial_points)
            score = 1-np.mean(rmse)
            return score
        else:
            return pred_output, score

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

    def iterate(self, raw_input_data):
        """ Iterates the reservoir and return all the successive reservoir states """
        if len(raw_input_data.shape) == 2:
            # if input data is 2D (previous pred_output) means it is in refreshing phase
            raw_input_data = self.encode_input(raw_input_data)
            input_data = raw_input_data.reshape((raw_input_data.shape[0], 1, raw_input_data.shape[1]))
            self.forget = 0
            self.parallel_runs = input_data.shape[0]
        else:
            input_data = self.encode_input(raw_input_data)
        n_sequence, sequence_length, input_dim = input_data.shape
        act = self.activation()
        res_states = np.zeros((n_sequence, sequence_length - self.forget, self.n_res), dtype=np.complex64)

        # Initialize hardware if we use the optical setup
        if self.random_projection == 'lighton opu':
            self.opu.open()
        elif self.random_projection == 'meadowlark slm':
            if self.matlab_eng is None:
                import matlab.engine
                self.matlab_eng = matlab.engine.start_matlab()
                self.matlab_eng.cd(r'D:\Users\Comedia\Desktop\reservoir-computing-python\hardware_control', nargout=0)
                # camera initialization
                if self.cam_roi is None:
                    self.cam_roi = [350, 350]
                self.matlab_eng.workspace['cam_roi'] = matlab.double(self.cam_roi)
                self.matlab_eng.camera_open(nargout=0)
                self.cam_sampling_range = np.linspace(0, (self.cam_roi[0]-1)*(self.cam_roi[1]-1)-1, self.n_res, dtype='uint32')
                if self.n_res > (self.cam_roi[0]-1)*(self.cam_roi[1]-1)-1:
                    warnings.warn("The number of camera pixels is less than the required size of the reservoir")
                # SLM initialization
                if self.slm_size is None:
                    self.slm_size = [512, 512]
                self.matlab_eng.workspace['slm_size'] = matlab.double(self.slm_size)
                self.matlab_eng.slm_open(nargout=0)

        for i_sequence in range(int(n_sequence / self.parallel_runs)):
            idx_sequence = np.arange(i_sequence * self.parallel_runs, (i_sequence + 1) * self.parallel_runs)
            if self.verbose and sequence_length>1:
                time_iterable = tqdm_notebook(range(sequence_length), file=sys.stdout,  desc='reservoir construction')
            else:
                time_iterable = range(sequence_length)
            for time_step in time_iterable:
                if self.random_projection == 'simulation':
                    self.state = act(
                        np.dot(self.input_w, input_data[idx_sequence, time_step, :].T) +
                        self.leak_rate * np.dot(self.res_w, self.encode_res(self.state.T).T) + 
                        (1 - self.leak_rate) * self.state + 
                        self.bias_vec)
                elif self.random_projection == 'hyperdimensional':
                    # Remove the reservoir weights
                    previous_state = self.state
                    self.state = self.encode_res(self.state)
                    self.state = self.leak_rate * act(np.dot(self.input_w, input_data[idx_sequence, time_step, :].T) +
                                    + self.bias_vec)+ \
                                 (1 - self.leak_rate) * np.roll(previous_state, 1, axis=0)
                elif self.random_projection == 'lighton opu':
                    # Create image from self.state and input_data
                    state_img = self.encode_res(self.state).astype(np.uint8)

                    # if any(self.state.flatten() > 35):
                    #     state_img = (self.state > 35).astype(np.uint8)
                    # else:
                    #     state_img = self.state

                    # state_img = encode.large_bin_binary(self.state, 0, 200, 10, 0.5).astype(np.uint8)
                    # state_img  = (self.state > 65).astype(np.uint8)
                    current_input_data = input_data[idx_sequence, time_step, :].astype(np.uint8)

                    # Previous code that might be relevant
                    # if len(input_data.shape) == 1:
                    #     input_data = input_data.reshape((1, -1))
                    # if len(reservoir.shape) == 1:
                    #     reservoir = reservoir.reshape((-1, 1))

                    res_repeat = int(np.round(self.res_scale ** 2))
                    res_size = int(res_repeat * self.n_res * self.res_enc_dim)

                    input_repeat = int(np.round(self.n_res * self.res_enc_dim * self.input_scale ** 2 / self.input_dim / self.input_enc_dim))
                    input_size = int(input_repeat * input_dim)

                    bias_repeat = int(np.round(self.n_res * self.res_enc_dim * self.bias_scale ** 2))
                    bias_size = int(bias_repeat)

                    total_size = np.int(res_size + input_size + bias_size)
                    slm_imgs = np.zeros((n_sequence, total_size))
                    slm_imgs[:, :bias_repeat] = np.ones((n_sequence, bias_repeat))
                    slm_imgs[:, bias_repeat:bias_repeat+res_size] = np.repeat(state_img.T, res_repeat, axis=1)
                    slm_imgs[:, bias_repeat+res_size:] = np.repeat(current_input_data, input_repeat, axis=1)
                    # slm_imgs[:, :res_size] = np.repeat(state_img.T, res_repeat, axis=1)
                    # slm_imgs[:, res_size:res_size + input_size] = np.repeat(current_input_data, input_repeat, axis=1)
                    # slm_imgs[:, res_size+input_size:] = np.ones((n_sequence, bias_repeat))

                    # import matplotlib.pyplot as plt
                    # plt.imshow(slm_imgs[0, :].reshape(30, -1))

                    cam_img = self.random_mapping.fit_transform(slm_imgs)
                    self.state = self.leak_rate * \
                                 (np.sqrt(cam_img.T) * 16).astype(np.uint8) + \
                                 (1 - self.leak_rate) * self.state
                    # self.state = (self.random_mapping.fit_transform(slm_imgs).T > 35).astype(np.uint8)

                    # Display of DMD and camera image
                    formatted_dmd_img = self.random_mapping.formatting_func(slm_imgs[0, :].reshape(1, -1))
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

                    ax1.imshow(self.state[:, 0].reshape(23, -1))
                    ax2.imshow(formatted_dmd_img.reshape(1140, 912), cmap='gray')
                    self.img_dmd = formatted_dmd_img.reshape(1140, 912)
                    self.img_cam = cam_img
                    plt.show()
                elif self.random_projection == 'meadowlark slm':
                    # print('state.shape_before = '+str(self.state.shape))
                    self.state = self.encode_res(self.state)
                    # print('state.shape_after = '+str(self.state.shape))
                    # print('input_data.shape = '+str(input_data.shape))
                    if self.parallel_runs==1 and self.state.shape[1] == 1:
                        slm_imgs = self.generate_slm_imgs(input_data[idx_sequence, time_step, :], self.state.T)
                        adict = {}
                        adict['phase_vec'] = np.array(slm_imgs[0,:], dtype='uint8') # since SLM is 8bit
                        sio.savemat('hardware_control/phase_vec.mat', adict)
                        self.matlab_eng.get_speckle(nargout=0)
                        cam_data_matlab = self.matlab_eng.workspace['data']
                        self.state = ((1-self.leak_rate)*np.ravel(np.array(cam_data_matlab._data).reshape(
                            cam_data_matlab.size[::-1]).T)[self.cam_sampling_range] + self.leak_rate*self.state.T).reshape(-1, 1)
                        # print('__'+str(self.state.shape))
                    else:
                        selfstate = np.zeros((self.n_res, self.state.shape[1]))
                        for i in range(self.state.shape[1]):
                            # print('_____'+str(self.state.shape))
                            state = self.state[:, i].reshape(-1, 1)
                            # print('state.shape = '+str(state.shape))
                            slm_imgs = self.generate_slm_imgs(input_data[i, :, :], state.T)
                            adict = {}
                            adict['phase_vec'] = np.array(slm_imgs[0,:], dtype='uint8') # since SLM is 8bit
                            sio.savemat('hardware_control/phase_vec.mat', adict)
                            self.matlab_eng.get_speckle(nargout=0)
                            cam_data_matlab = self.matlab_eng.workspace['data']
                            # print(state.T.shape)
                            # print('np.array(cam_data_matlab._data).reshape(cam_data_matlab.size[::-1]).T'+str(np.array(cam_data_matlab._data).reshape(cam_data_matlab.size[::-1]).T.shape))
                            # print(np.ravel(np.array(cam_data_matlab._data).reshape(cam_data_matlab.size[::-1]).T)[self.cam_sampling_range].shape)
                            selfstate[:, i] = ((1-self.leak_rate)*np.ravel(np.array(cam_data_matlab._data).reshape(
                                cam_data_matlab.size[::-1]).T)[self.cam_sampling_range] + self.leak_rate*state.reshape(-1))
                        self.state = selfstate

                if time_step >= self.forget:
                    res_states[idx_sequence, time_step - self.forget, :] = self.state.T
        res_states = np.real_if_close(res_states)
        self.state = res_states.reshape((-1, res_states.shape[-1])).T # will be used in update equation if recursive prediction is active
        # print('self.state = '+str(self.state.shape))
        state_iscomplex = True if any(np.iscomplex(res_states.flatten())) else False
        # print('res_states'+str(res_states.shape))
        res_states = np.concatenate((np.abs(res_states) ** 2,
                                     np.angle(res_states, deg=False))) if state_iscomplex else res_states

        # standardization of the reservoir
        if self.res_standardize:
            for i in range(n_sequence):
                preprocessing.scale(res_states[i, :, :], axis=0, copy=False)
        if self.scale_res_MinMax:
            encode.scale(res_states, self.scale_res_MinMax, in_place=True)

        # construct the concatenated states
        concat_states = res_states
        # print('res_states'+str(res_states.shape))
        if self.raw_input_feature:
            concat_states = np.concatenate((concat_states, raw_input_data[:, self.forget:, :]), axis=2)
        if self.enc_input_feature:
            enc_input_iscomplex = True if any(np.iscomplex(input_data.flatten())) else False
            if enc_input_iscomplex:
                print('enc_input_iscomplex')
            enc_input_data = np.angle(
                input_data, deg=False) if enc_input_iscomplex else input_data
            # print('enc_input_data[:, self.forget:, :]='+str(enc_input_data[:, self.forget:, :].shape))
            # print('concat_states'+str(concat_states.shape))
            # print(enc_input_data[:, self.forget:, :])
            concat_states = np.concatenate((concat_states, enc_input_data[:, self.forget:, :]), axis=2)
            # print(concat_states)

        if self.parallel_res > 1:
            n_sequence, sequence_length, concat_dim = concat_states.shape
            concat_states = concat_states.reshape((
                int(n_sequence/self.parallel_res), self.parallel_res, sequence_length, concat_dim)).transpose(
                (0, 2, 1, 3)).reshape((int(n_sequence/self.parallel_res), sequence_length, self.parallel_res*concat_dim))

        # Release hardware if we use the optical setup
        if self.random_projection == 'lighton opu':
            self.opu.close()
#         elif self.random_projection == 'meadowlark slm':
#             self.matlab_eng.close_all(nargout=0)
#             self.matlab_eng = None

        if self.verbose >= 2:
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
        self.forget = self.forget__
        self.parallel_runs = self.parallel_runs__
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
        elif self.train_method == 'ridge_parallel':
            # because of processors overheating can work longer than the standard ridg regression
            from sklearn.multioutput import MultiOutputRegressor
            clf = MultiOutputRegressor(sklearn.linear_model.Ridge(fit_intercept=False, alpha=self.train_param), n_jobs=3)
            clf.fit(concat_states, y)
            return np.array([clf.estimators_[i].coef_ for i in range(len(clf.estimators_))]).T

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

    @staticmethod
    def score_metric(pred_output, output):
        #return 1 - np.sum(np.abs(pred_output-output)**2) / max(np.sum(np.abs(output-np.mean(output))**2),
        #                                                       np.sum(np.abs(pred_output - np.mean(pred_output)) ** 2))
        return np.abs(np.sum(np.conj(pred_output)*(output)))**2 / (np.linalg.norm(pred_output)*np.linalg.norm(output))**2
    # Correlation
    # 1 - np.abs(np.conj(pred_output).dot(output))**2 / (np.linalg.norm(pred_output)*np.linalg.norm(output))**2
    # Or do not add the center in the denominator

    def detailed_pred_score(self, pred_output, true_data, spatial_points):
        # print('pred_output = '+str(pred_output[:3, :4]))
        # print('true_data = '+str(true_data[:3, :4]))
        if self.future_pred is False:
            print('The "detailed_pred_score" function should only be called in prediction mode.')
            return -1
        # score_vec = 1 - np.sum(np.abs(pred_output - output)**2, axis=0) / \
            # np.sum(np.abs(output - np.mean(output, axis=0))**2, axis=0)

        total_pred = self.pred_horizon*self.rec_pred_steps
        # print('total_pred = '+str(total_pred))
        true_data_std = np.std(true_data) # think about better normalization
        true_data_norm = true_data/true_data_std
        pred_output_norm = pred_output/true_data_std
        length_input = pred_output.shape[0]
        rand = np.random.rand(pred_output_norm.shape[0], pred_output_norm.shape[1])*max(abs(true_data_norm.flatten()))

        rmse = np.zeros((length_input-1, total_pred-1))
        rmse_rand = np.zeros((length_input-1, total_pred-1))
        for n_input in range(1, length_input):
            for n_pred in range(1, total_pred):
                d1 = pred_output_norm[n_input, :n_pred*spatial_points]
                d2 = true_data_norm[n_input, :n_pred*spatial_points]
                d_rand = rand[n_input, :n_pred*spatial_points]
                rmse[n_input-1, n_pred-1] = np.sqrt(1./(n_pred*spatial_points)*np.sum((d1.flatten() - d2.flatten())**2))
                rmse_rand[n_input-1, n_pred-1] = np.sqrt(1./(n_pred*spatial_points)*np.sum((d_rand.flatten() - d2.flatten())**2))
        norm = np.mean(rmse_rand)
        rmse = rmse / norm
        rmse_vec = np.mean(rmse, axis=0)
        rmse_vert = np.mean(rmse, axis=1)
        return rmse, rmse_vec, rmse_vert

    def generate_slm_imgs(self, input_data, reservoir):
        # We first fix the size of the reservoir
        # print('input_data.shape = '+str(input_data.shape))
        # reservoir = reservoir.T
        # print('reservoir.shape = '+str(reservoir.shape))
        res_repeat = np.round(self.res_scale**2)
        res_size = res_repeat * self.n_res

        # We find how many times to repeat the input
        n_sequence, input_dim = input_data.shape
        input_repeat = np.round(self.n_res * self.input_scale**2 / input_dim)
        input_size = int(input_repeat * input_dim)

        # We find the bias
        bias_repeat = np.round(self.n_res * self.bias_scale**2)
        bias_size = bias_repeat

        # We put everything in a new vector
        total_size = np.int(res_size + input_size + bias_size)

        slm_imgs = np.zeros((n_sequence, total_size))
        # print('slm_imgs[:, :res_size] = '+str(slm_imgs[:, :res_size].shape))
        # print('np.repeat(reservoir, res_repeat, axis=1).shape = '+str(np.repeat(reservoir, res_repeat, axis=1).shape))
        slm_imgs[:, :res_size] = np.repeat(reservoir, res_repeat, axis=1)
        slm_imgs[:, res_size:res_size+input_size] = np.repeat(input_data, input_repeat, axis=1)
        slm_imgs[:, res_size+input_size:] = np.repeat([[1]]*n_sequence, bias_repeat, axis=1)

        return slm_imgs
        