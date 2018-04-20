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
    input_data, y = data.mackey_glass(sequence_length=300)
    b = Reservoir(n_res=10000, input_scale=2, train_method='ridge',
    	weights_type='complex gaussian', random_projection='simulation',
    	activation_fun='binary', activation_param=2,
    	encoding_method='realbinary', n_input = 1000,
        forget=50)
    print(b)

    start = time.time()
    b.fit(input_data, y)
    train_score = b.fit_score

    input_data, y = data.mackey_glass(sequence_length=200)
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
    # with open('out/true.out', 'w') as f:
        # print(np.ravel(y[b.forget:, :]), file=f)
