#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import sys
# import numpy as np

# import time

from reservoir import Reservoir
import data

if __name__ == "__main__":
    print('Hello world!')
    input_data, y = data.narma()
    b = Reservoir(n_res=500)
    print(b)
    b.fit(input_data, y)
    print(b.score(input_data, y[b.forget:]))
    # b.initialize()
    # test = b.iterate(input_data)
    # print(test)
