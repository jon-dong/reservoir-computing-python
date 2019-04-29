"""
Reservoir class using GPU, a custom sklearn estimator for Reservoir Computing.

Methods:
    - TODO

Internal methods:
    - TODO
Parameters:
    - TODO

Internal attributes:
    - TODO

To-do:
    -
"""

import cupy as np

from sklearn.utils import check_random_state
import time
from tqdm import tqdm
import sys
import data_utils
import scipy.io as sio


class Reservoir(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_res=400, res_scale=1,  # reservoir
                 input_dim=None, input_scale=1,  # input
                 add_bias=True, bias_scale=1,  # bias
                 weights_type='gaussian', activation_fun='tanh', leak_rate=1,  # dynamics
                 parallel_runs=None, forget=100,  # iterations
                 train_param=1e1,  # fit
                 random_state=None, save=0, verbose=1):  # misc
        self.n_res = n_res
        self.res_scale = res_scale
        self.input_dim = input_dim
        self.input_scale = input_scale
        self.add_bias = add_bias
        self.bias_scale = bias_scale
        self.weights_type = weights_type
        self.activation_fun = activation_fun
        self.leak_rate = leak_rate
        self.parallel_runs = parallel_runs
        self.forget = forget
        self.train_param = train_param
        self.random_state = random_state
        self.save = save
        self.verbose = verbose

        self.input_w = None
        self.res_w = None
        self.output_w = None
        self.state = None

        self.fit_score = None
        self.init_timer = None
        self.iterate_timer = None
        self.train_timer = None

    def fit(self, input_data, y=None):
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

    def score(self, input_data, true_output=None, sample_weight=None):
        return self.predict_and_score(input_data, true_output, only_score=True)

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
        true_output = true_output[:, self.forget:, :]

        if self.future_pred and detailed_score:
            # print(pred_output.shape)
            # print(true_output.shape)
            score = self.detailed_pred_score(pred_output, true_output)
        else:
            true_output = true_output.reshape(-1, true_output.shape[-1])
            pred_output = pred_output.reshape(-1, pred_output.shape[-1])
            score = self.score_metric(pred_output, true_output)
        # plt.plot(pred_output[0, :, 0])
        # plt.plot(true_output[0, :, 0])
        # plt.show()
        true_output = true_output.reshape(-1, true_output.shape[-1])
        pred_output = pred_output.reshape(-1, pred_output.shape[-1])
        final_score = self.score_metric(pred_output, true_output)

        test_end = time.time()
        test_timer = test_end - iterate_end
        if self.verbose:
            print('Testing complete. \t\t\tElapsed time: ' + str(test_timer) + ' s')
            print('Testing score: ' + str(final_score))

        if only_score:
            return score
        else:
            return pred_output, score

    def initialize(self):
        """ Initializes the reservoir state, the input and reservoir weights """
        self.random_state = check_random_state(self.random_state)
        total_input_dim = self.input_dim * self.input_enc_dim
        total_res_dim = self.n_res * self.res_enc_dim
        self.bias_vec = self.add_bias * self.random_state.normal(loc=0., scale=self.bias_scale,
                                                                 size=(self.n_res, 1))
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

    def reset_state(self):
        """ Resets the reservoir state, for new runs """
        # To-do: add different statistics
        if self.parallel_runs is None:
            self.state = self.random_state.normal(loc=0., scale=1, size=self.n_res)
        else:
            self.state = self.random_state.normal(loc=0., scale=1, size=(self.n_res, self.parallel_runs))
            # self.state = self.random_state.randint(0, 100, size=(self.n_res, self.parallel_runs), dtype=np.uint8)

    def encode_input(self, mat):
        return mat

    def encode_res(self, mat):
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
        # state_iscomplex = True if self.activation_fun == 'phase' or self.activation_fun == 'phase 8bit' else False
        n_parallel = self.parallel_runs if self.parallel_runs is not None else 1
        concat_states = np.zeros((n_sequence, sequence_length - self.forget,
                                  self.n_res + input_dim), dtype=np.float64)
        act = self.activation()

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
                self.state = self.leak_rate * act(
                    np.dot(self.input_w, input_data[idx_sequence, time_step, :].T) +
                    np.dot(self.res_w, self.encode_res(self.state)) + self.bias_vec) + \
                             (1 - self.leak_rate) * self.state

                if time_step >= self.forget:
                    state = np.angle(self.state, deg=False) if any(np.iscomplex(self.state.flatten())) else self.state
                    inputdata = np.angle(input_data[idx_sequence, time_step, :], deg=False) \
                        if input_iscomplex else input_data[idx_sequence, time_step, :]
                    concat_states[idx_sequence, time_step - self.forget, :] = np.concatenate((state, inputdata.T)).T

        if self.verbose >= 2:
            res_states = concat_states[:, :, :self.n_res]
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
        concat_states = concat_states.reshape(-1, concat_states.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        d = y.shape[-1]

        import torch
        from torch.autograd import Variable
        from torch.nn import Linear, Module, MSELoss
        from torch.optim import LBFGS
        from torch.utils.dlpack import to_dlpack
        from torch.utils.dlpack import from_dlpack
        class LinearRegressionModel(torch.nn.Module):
            def __init__(self, d=100):
                super(LinearRegressionModel, self).__init__()
                self.fc = Linear(d, 1)

            def forward(self, x):
                x = self.fc(x)
                return x

        concat_states = from_dlpack(concat_states.toDlpack()).type(torch.cuda.FloatTensor)
        y = from_dlpack(y.toDlpack()).type(torch.cuda.FloatTensor)

        model = LinearRegressionModel(d=d).cuda()
        # define criterion - loss function
        criterion = MSELoss()
        # define optimizer
        optimizer = LBFGS(model.parameters(), max_iter=10, history_size=5)

        def closure():
            y_pred = model(Variable(concat_states))
            #             print(y_pred.shape)
            #             print(err.shape)
            loss = criterion(y_pred, Variable(y))
            epoch_loss = loss.data
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)
        return np.fromDlpack(to_dlpack(model.fc.weight.data)).T

    def output(self, concat_states):
        """ Computes the output given reservoir states and output weights """
        total_output = np.dot(concat_states, self.output_w)
        return total_output

    @staticmethod
    def score_metric(pred_output, output):
        # return 1 - np.sum(np.abs(pred_output-output)**2) / max(np.sum(np.abs(output-np.mean(output))**2),
        #        np.sum(np.abs(pred_output - np.mean(pred_output)) ** 2))
        return np.abs(np.sum(np.conj(pred_output) * output)) ** 2 / (
                    np.linalg.norm(pred_output) * np.linalg.norm(output)) ** 2

    # Correlation
    # 1 - np.abs(np.conj(pred_output).dot(output))**2 / (np.linalg.norm(pred_output)*np.linalg.norm(output))**2
    # Or do not add the center in the denominator

    def detailed_pred_score(self, pred_output, output):
        if self.future_pred is False:
            print('The "detailed_pred_score" function should only be called in prediction mode.')
            return -1
        # score_vec = 1 - np.sum(np.abs(pred_output - output)**2, axis=0) / \
        # np.sum(np.abs(output - np.mean(output, axis=0))**2, axis=0)

        n_parallel, sequence_length, pred_horizon = output.shape
        effective_length = sequence_length - pred_horizon  # remove the end to avoid aliasing effects
        rmse = np.zeros((effective_length, pred_horizon))
        for t in range(effective_length):
            for pred in range(pred_horizon):
                d1 = pred_output[:, t, 0:pred].flatten()  # for 1D time series
                d2 = output[:, t, 0:pred].flatten()
                rmse[t, pred] = np.sqrt(np.sum((d1 - d2) ** 2) / (pred + 1) / n_parallel)
        score_vec = np.mean(rmse, axis=0)

        return score_vec
