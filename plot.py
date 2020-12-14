# from pylab import *
# import matplotlib.pyplot as pyplot
# a = [ pow(10,i) for i in range(10) ]
# fig = pyplot.figure()
# ax = fig.add_subplot(2,1,1)
# line, = ax.plot(a, color='blue', lw=2)
# show()

import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

N = 96
K = 48
BP_iter_num = 5

list_BP_iter_num = [10, 20, 40, 50, 60]
# 线条配合形状，构建6种不同图例
list_marker = ["", "", "", "", "o", "."]
list_line_style = ['-', '--', '-.', ':', '-', '-']

for i in range(5):
    BP_iter_num = list_BP_iter_num[i]
    marker = list_marker[i]
    linestyle = list_line_style[i]
    plot_file = format("model/data_back_up/BER(%s_%s)_BP(%s).txt" % (N, K, BP_iter_num))

    plot_data = np.loadtxt(plot_file, dtype=np.float32)
    x = plot_data[:, 0]
    y = plot_data[:, 1]
    label = format("(%s,%s)iter=%s" % (N, K, BP_iter_num))
    plt.semilogy(x, y, label=label, marker=marker, linestyle=linestyle)

plt.grid(True, which="both", ls="-")
plt.xlabel("SNR")
plt.ylabel("BER")
plt.legend()  # 启用图例（还可以设置图例的位置等等）

plt.show()