import numpy as np
import matplotlib.pyplot as plt

pred = np.loadtxt("rc8/out/predict.txt", dtype=float)
y = np.loadtxt("rc8/out/true.txt", dtype=float)

plt.plot(y[10::101], 'r--', linewidth=2)
plt.plot(pred[10::101], 'b', linewidth=1)

plt.xticks(fontsize=14)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Prediction', fontsize=18)

plt.show()
 
 