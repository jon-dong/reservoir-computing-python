#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import sys
import numpy as np

import time

from reservoir import Reservoir
import data

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    params = np.logspace(1e4, 4e4, 5)
    n_params = len(params)
    n_repeat = 1

    times = np.empty((n_repeat, n_params))
    results = np.empty((n_repeat, n_params))

    i_param = 0
    for param in params:
        for i_repeat in range(n_repeat):
            try:
                input_data, y = data.mackey_glass(sequence_length=2000)
                b = Reservoir(n_res=int(param), input_scale=2, train_method='ridge',
                	weights_type='complex gaussian', random_projection='simulation',
                	activation_fun='binary', activation_param=2,
                	encoding_method='realbinary', n_input = 1000)
                print(b)

                start = time.time()
                b.fit(input_data, y)
                end = time.time()
                print(end - start)

                input_data, y = data.mackey_glass()
                current_score = b.score(input_data, y[b.forget:])
                print(current_score)

                times[i_repeat, i_param] = end - start
                results[i_repeat, i_param] = current_score
            except:
                times[i_repeat, i_param] = None
                results[i_repeat, i_param] = None

        with open('out/times.out', 'w') as f:
            print(times, file=f)
        with open('out/results.out', 'w') as f:
            print(results, file=f)
        with open('out/params.out', 'w') as f:
            print(params, file=f)

        i_param += 1
