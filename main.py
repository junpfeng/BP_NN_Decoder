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
'''
更改码字的时候，需要配套修改的类有：
1. top_config中的N和K 、 matlab 中生成噪声的部分和 matlab 中生成生成矩阵和校验矩阵部分：
2. 是否需要启用训练的BP：BP_Decoder.py 中的self.use_train_bp_net ；是否需要启用conv net：net_config中的self.use_conv_net
    2.1 如果不需要使用 cnn,则将 top_config 中的 cnn_net_number 设为 0，同时将 BP_iter_nums_simu 设为只有一个元素。
3. 当启用新的码字时，先运行 train_bp_network 建立并训练对应的BP网络。
4. 如果恢复网络参数时,发生错误,提示网络不匹配,则有可能是batch_size不匹配,尝试删除已有的网络,并修改batch_size.
5. conv 和 bp 不能一起训练，bp 在获取session 的时候，会将 conv 的session获取过来，从而导致存储了双图。
'''
batch_size = int(train_config.training_minibatch_size // np.size(train_config.SNR_set_gen_data))
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
elif top_config.function == 'TrainConv':
    net_id = top_config.currently_trained_net_id
    # 定义一个卷积网络对象
    conv_net = ConvNet.ConvNet(net_config, train_config, net_id)
    # 开始训练网络
    conv_net.train_network(top_config.model_id)

elif top_config.function == 'TrainBP':
    # 训练 BP 网络
    ibd.train_bp_network(code, top_config, net_config, batch_size)

elif top_config.function == 'Simulation':
    # if top_config.analyze_res_noise:  # 分析残差噪声
    #     simutimes_for_anal_res_power = int(np.ceil(5e6 / float(top_config.K_code * batch_size)) * batch_size)
    #     ibd.analyze_residual_noise(code, top_config, net_config, simutimes_for_anal_res_power, batch_size)
    simutimes_range = np.array([np.ceil(1e7 / float(top_config.K_code * batch_size)) * batch_size, np.ceil(1e8 / float(top_config.K_code * batch_size)) * batch_size], np.int32)
    ibd.simulation_colored_noise(code, top_config, net_config, simutimes_range, 1000, batch_size)
