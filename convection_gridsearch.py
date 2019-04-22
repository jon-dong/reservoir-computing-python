#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from reservoir import Reservoir
import data_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def load_input_33(path, n_repeat=1):
    T_3D = np.load(path)[1:-1, :, :] # skipping the boundary layers
    x_dim, y_dim, sequence_length = T_3D.shape
    T_2D = T_3D.reshape((x_dim*y_dim, sequence_length)).T.reshape((1, sequence_length, x_dim*y_dim))
    T_2D_norm = (T_2D - (np.amax(T_2D) + np.amin(T_2D))/2)/((np.amax(T_2D) - np.amin(T_2D))/2)
    T_2D_norm = np.tile(T_2D_norm, (n_repeat, 1, 1))
    return T_3D, T_2D_norm

_, train_data_2 = load_input_33("2D_convection_datasets/x_y_temperature_deltaT_2.npy", n_repeat = 1)
_, train_data_4 = load_input_33("2D_convection_datasets/x_y_temperature_deltaT_4.npy", n_repeat = 1)
_, train_data_6 = load_input_33("2D_convection_datasets/x_y_temperature_deltaT_6.npy", n_repeat = 1)
_, train_data_8 = load_input_33("2D_convection_datasets/x_y_temperature_deltaT_8.npy", n_repeat = 1)
_, train_data_10 = load_input_33("2D_convection_datasets/x_y_temperature_deltaT_10.npy", n_repeat = 1)
train_data = np.concatenate((train_data_2, train_data_4, train_data_6, train_data_8, train_data_10))

n_sequence, sequence_length, spatial_points = train_data.shape

b = Reservoir(n_res=2000, res_scale=1, res_encoding='phase', res_enc_param=1.5*np.pi,
              input_scale=1, input_encoding='phase', input_enc_param = 1.5*np.pi,
              random_projection='simulation', weights_type='complex gaussian',
              activation_fun='intensity', activation_param=10,
              parallel_runs=n_sequence,  bias_scale=0.2, leak_rate=0.15,
              pred_horizon=1, rec_pred_steps=1, forget = 50,
              train_method='ridge', train_param=1e3, verbose=1
             )
params = [
    {'res_scale': np.array([.5, 1, 2]),
     'input_scale': np.array([.5, 1, 2]),
     'bias_scale': np.array([0.1, 0.5, 1]),
     'train_param': np.array([1e-2, 1e2, 1e4]),
     'leak_rate': np.array([0.1, 0.3]),
     'activation_param': np.array([1, 10])}
]
grid_search = GridSearchCV(estimator=b, param_grid=params, return_train_score=True, cv=3, verbose=2, n_jobs=6)
grid_search.fit(train_data)

print(grid_search.best_params_)