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

# from lightonml.random_projections.opu import OPURandomMapping
# from lightonopu.opu import OPU

if __name__ == "__main__":
    # params = [1e2, 2e2, 3e2, 4e2, 6e2, 8e2, 1.2e3, 1.6e3, 2.4e3, 3.2e3, 6.4e3, 1.28e4]
    params = [1e2, 2e2]
    n_params = len(params)
    n_repeat = 10

    times = np.empty((n_repeat, n_params))
    results = np.empty((n_repeat, n_params))

    i_param = 0
    for param in params:
        for i_repeat in range(n_repeat):
            input_data, y = data.mackey_glass(sequence_length=2000)
            b = Reservoir(n_res=1000, input_scale=2, train_method='ridge',
                          weights_type='complex gaussian', random_projection='simulation',
                          activation_fun='binary', activation_param=1,
                          encoding_method='naivebinary', input_dim=1000, forget=100)
            # n_sequence, sequence_length, input_dim
            print(b)

            start = time.time()
            b.fit(input_data, y)
            train_score = b.fit_score

            input_data, y = data.mackey_glass(sequence_length=1000)
            valid_score = b.score(input_data, np.ravel(y[b.forget:]))
            end = time.time()
            print('True output')
            print(np.ravel(y[b.forget:]))
            print('Elapsed time')
            print(end - start)
            print('Iterate time')
            print(b.iterate_timer)
            print('Fit time')
            print(b.train_timer)
            print('Train score')
            print(train_score)
            print('Validation score')
            print(valid_score)

            np.savetxt('out/true.txt', np.ravel(y[b.forget:]), fmt='%f')

            times[i_repeat, i_param] = b.iterate_timer
            results[i_repeat, i_param] = valid_score
            # try:
            #     input_data, y = data.mackey_glass(sequence_length=2000)
            #     b = Reservoir(n_res=1000, input_scale=2, train_method='ridge',
            #     	weights_type='complex gaussian', random_projection='simulation',
            #     	activation_fun='binary', activation_param=1,
            #     	encoding_method='realbinary', n_input = 1000,
            #         forget=100)
            #     print(b)
            #
            #     start = time.time()
            #     b.fit(input_data, y)
            #     train_score = b.fit_score
            #
            #     input_data, y = data.mackey_glass(sequence_length=1000)
            #     valid_score = b.score(input_data, np.ravel(y[b.forget:]))
            #     end = time.time()
            #     print('True output')
            #     print(np.ravel(y[b.forget:]))
            #     print('Elapsed time')
            #     print(end - start)
            #     print('Iterate time')
            #     print(b.iterate_timer)
            #     print('Fit time')
            #     print(b.train_timer)
            #     print('Train score')
            #     print(train_score)
            #     print('Validation score')
            #     print(valid_score)
            #
            #     np.savetxt('out/true.txt', np.ravel(y[b.forget:]), fmt='%f')
            #
            #     times[i_repeat, i_param] = b.iterate_timer
            #     results[i_repeat, i_param] = valid_score
            # except:
            #     times[i_repeat, i_param] = None
            #     results[i_repeat, i_param] = None

        i_param += 1

    with open('out/times.out', 'w') as f:
        print(times, file=f)
    with open('out/results.out', 'w') as f:
        print(results, file=f)
    with open('out/params.out', 'w') as f:
        print(params, file=f)
