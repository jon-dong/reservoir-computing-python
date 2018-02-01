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
    input_data, y = data.mackey_glass()
    b = Reservoir(n_res=1000, input_scale=2, train_method='sgd',
    	weights_type='complex gaussian', random_projection='out of core',
    	activation_fun='binary', activation_param=.4, 
    	encoding_method='realbinary', n_input = 1000)
    print(b)

    start = time.time()
    b.fit(input_data, y)
    end = time.time()
    print(end-start)

    print(b.score(input_data, y[b.forget:]))

    input_data, y = data.mackey_glass()
    n_plot = 500
    sns.set_style("darkgrid")
    if n_plot == -1:
        plt.plot(y[b.forget:])
        plt.plot(b.predict(input_data[:]))
    elif n_plot>0:
    	plt.plot(y[b.forget:b.forget+n_plot])
    	plt.plot(b.predict(input_data[:n_plot+100]))
    if n_plot != 0:
    	plt.show()


    # b.initialize()
    # test = b.iterate(input_data)
    # print(test)
