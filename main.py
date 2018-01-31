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
    b = Reservoir(n_res=500, train_method='explicit')
    print(b)

    start = time.time()
    b.fit(input_data, y)
    end = time.time()
    print(end-start)

    print(b.score(input_data, y[b.forget:]))

    sns.set_style("darkgrid")
    plt.plot(input_data[b.forget:])
    plt.plot(b.predict(input_data))
    plt.show()


    # b.initialize()
    # test = b.iterate(input_data)
    # print(test)
