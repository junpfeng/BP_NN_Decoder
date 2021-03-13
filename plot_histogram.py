# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 仿真时间统一换算到 100 w个码字

name_list = ['BPDNN25', 'advance_BPDNN20']
# name_list = ['BPDNN25', 'LLRBP50']
num_list = [767, 892]
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list, width=0.2)
plt.ylabel("time/s")
plt.title('LDPC(576, 432) decode simulation time')

plt.legend()  # 启用图例（还可以设置图例的位置等等）
plt.show()