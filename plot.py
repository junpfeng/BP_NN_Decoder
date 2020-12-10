# from pylab import *
# import matplotlib.pyplot as pyplot
# a = [ pow(10,i) for i in range(10) ]
# fig = pyplot.figure()
# ax = fig.add_subplot(2,1,1)
# line, = ax.plot(a, color='blue', lw=2)
# show()

import numpy as np
from matplotlib import pyplot as plt

N = 16
K = 8

plot_file = format("model/BER(%s_%s)_BP(5_5)_model0.txt" % (N, K))

plot_data = np.loadtxt(plot_file, dtype=np.float32)
x = plot_data[:, 0]
y = plot_data[:, 1]

plt.semilogy(x,y)
plt.grid(True,which="both",ls="-")
plt.show()