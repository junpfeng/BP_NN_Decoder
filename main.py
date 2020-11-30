# coding=utf-8
#  #################################################################
#  Python code to reproduce our works on iterative BP-CNN.
#
#  Codes have been tested successfully on Python 3.4.5 with TensorFlow 1.1.0.
#
#  References:
#   Fei Liang, Cong Shen and Feng Wu, "An Iterative BP-CNN Architecture for Channel Decoding under Correlated Noise", IEEE JSTSP
#
#  Written by Fei Liang (lfbeyond@mail.ustc.edu.cn, liang.fei@outlook.com)
#  #################################################################


import sys
import Configrations
import numpy as np
import LinearBlkCodes as lbc
import Iterative_BP_CNN as ibd
import ConvNet
import DataIO

# address configurations
top_config = Configrations.TopConfig()
top_config.parse_cmd_line(sys.argv)

train_config = Configrations.TrainingConfig(top_config)
net_config = Configrations.NetConfig(top_config)

# (n,k)线性分组码，G:生成矩阵，H:校验矩阵， n 是经过生成矩阵之后的码长，k 是原来的码长
code = lbc.LDPC(top_config.N_code, top_config.K_code, top_config.file_G, top_config.file_H)

if top_config.function == 'GenData':
    # 定义一个噪声生成器，读取的噪声是 [ 576 * 576 ]
    noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file) # top_config.cov_1_2_file = Noise/cov_1_2_corr_para_0.5.dat
    # generate training data

    # code is LDPC object，产生训练数据
    ibd.generate_noise_samples(code, top_config, net_config, train_config, top_config.BP_iter_nums_gen_data,
                                                  top_config.currently_trained_net_id, 'Training', noise_io, top_config.model_id)
    # generate test data，产生测试数据集
    ibd.generate_noise_samples(code, top_config, net_config, train_config, top_config.BP_iter_nums_gen_data,
                                                  top_config.currently_trained_net_id, 'Test', noise_io, top_config.model_id)
elif top_config.function == 'Train':
    net_id = top_config.currently_trained_net_id
    # 定义一个卷积网络对象
    conv_net = ConvNet.ConvNet(net_config, train_config, net_id)
    # 开始训练网络
    conv_net.train_network(top_config.model_id)
elif top_config.function == 'Simulation':
    batch_size = 5000
    # if top_config.analyze_res_noise:  # 分析残差噪声
    #     simutimes_for_anal_res_power = int(np.ceil(5e6 / float(top_config.K_code * batch_size)) * batch_size)
    #     ibd.analyze_residual_noise(code, top_config, net_config, simutimes_for_anal_res_power, batch_size)

    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    ibd.simulation_colored_noise(code, top_config, net_config, simutimes_range, 1000, batch_size)
    # ibd.train_bp_network(code, top_config, net_config, simutimes_range, 1000, batch_size)