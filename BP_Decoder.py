import tensorflow as tf
import numpy as np
import os
import LinearBlkCodes as lbc
import DataIO
import Configrations

top_config = Configrations.TopConfig()


class GetMatrixForBPNet:
    # this class is to calculate the matrices used to perform BP process with matrix operation
    # test_H即校验矩阵，loc_nzero_row是校验矩阵中非零元素的坐标（横坐标和纵坐标分别存储）
    def __init__(self, test_H, loc_nzero_row):
        print("Construct the Matrics H class!\n")
        self.H = test_H
        self.m, self.n = np.shape(test_H)  # 校验矩阵是 144 行，576列！！！
        self.H_sum_line = np.sum(self.H, axis=0)  # 将校验矩阵每列相加(表示每列非零元素数量)，由（144, 576)变成 （1，576）,0-431元素都是4，432-455元素是3，456-575元素是2，总的1元素数量2040
        self.H_sum_row = np.sum(self.H, axis=1)  # 同上，每行相加(表示每列非零元素数量)，由(144, 576)变成（144，1）,其中72-95元素是15，其余元素都是14，于是，总的1的数量 =2040
        self.loc_nzero_row = loc_nzero_row
        self.num_all_edges = np.size(self.loc_nzero_row[1, :])  # 校验矩阵中所有1的元素数量是2040
        #  各种统计数据????
        self.loc_nzero1 = self.loc_nzero_row[1, :] * self.n + self.loc_nzero_row[0, :]  # 这种计算感觉是某种编码
        self.loc_nzero2 = np.sort(self.loc_nzero1)  # 进行排序
        self.loc_nzero_line = np.append([np.mod(self.loc_nzero2, self.n)], [self.loc_nzero2 // self.n], axis=0)  # 转为两行，第一行余数，第二行整除商，这个正好是edge的坐标（竖向排序）
        self.loc_nzero4 = self.loc_nzero_line[0, :] * self.n + self.loc_nzero_line[1, :]  # 余数 * 576 + 对应的商 正好是edge对应的位置（横向）
        self.loc_nzero5 = np.sort(self.loc_nzero4)
        # loc_nzero_line 内的是按找竖向顺序排列的非零元素的坐标
        # loc_nzero4 是非零元素的位置（比如（0，0）就是第0个非零元素，（0，1）就是第一个），元素排放顺序则是和loc_nzero_line 一致
        # loc_nzero5 则是单纯 loc_nzero4的排序


    ##########################################################################################################
    def get_Matrix_VC(self):
        H_x_to_xe0 = np.zeros([self.num_all_edges, self.n], np.float32)  # (2040, 576)
        H_sum_by_V_to_C = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)  # (2040, 2040)
        H_xe_last_to_y = np.zeros([self.n, self.num_all_edges], dtype=np.float32)  # (576, 2040)
        Map_row_to_line = np.zeros([self.num_all_edges, 1])  # (2040,1)

        for i in range(0, self.num_all_edges):
            Map_row_to_line[i] = np.where(self.loc_nzero1 == self.loc_nzero2[i])  # 返回loc_nzerol 中等于 loc_nzero2[i]的元素的索引
            # !!! Map_row_to_line 记录了 loc_nzero_row 到 loc_nzero_line 之间的映射关系，不过使用的是数组形式的哈希表
            # Map_row_to_line 即横向edge到纵向edge的更新矩阵
        map_H_row_to_line = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, self.num_all_edges):
            map_H_row_to_line[i, int(Map_row_to_line[i])] = 1

        count = 0
        for i in range(0, self.n):
            temp = count + self.H_sum_line[i]
            H_sum_by_V_to_C[count:temp, count:temp] = 1
            H_xe_last_to_y[i, count:temp] = 1
            H_x_to_xe0[count:temp, i] = 1
            for j in range(0, self.H_sum_line[i]):
                H_sum_by_V_to_C[count + j, count + j] = 0
            count = count + self.H_sum_line[i]
        print("return Matrics V-C successfully!\n")
        return H_x_to_xe0, np.matmul(H_sum_by_V_to_C, map_H_row_to_line), np.matmul(H_xe_last_to_y, map_H_row_to_line)
        # H_x_to_xe0 是码字到输入层变量节点的转换矩阵
        # H_sum_by_V_to_C 是纵向（校验节点到变量节点）的更新矩阵，map_H_row_to_line 是将横向edge转为纵向edge
        # H_xe_last_to_y　是最后一层的转换矩阵

    ###################################################################################################
    def  get_Matrix_CV(self):  # 获取从 check node -> variable node 的变换s矩阵

        H_sum_by_C_to_V = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)  # 2040 * 2040 的矩阵
        Map_line_to_row = np.zeros([self.num_all_edges, 1])  # 2040 * 1 的矩阵
        for i in range(0, self.num_all_edges):
            Map_line_to_row[i] = np.where(self.loc_nzero4 == self.loc_nzero5[i])
        map_H_line_to_row = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, np.size(self.loc_nzero1)):
            map_H_line_to_row[i, int(Map_line_to_row[i])] = 1

        count = 0
        for i in range(0, self.m):
            temp = count + self.H_sum_row[i]
            H_sum_by_C_to_V[count:temp, count:temp] = 1
            for j in range(0, self.H_sum_row[i]):
                H_sum_by_C_to_V[count + j, count + j] = 0
            count = count + self.H_sum_row[i]
        print("return Matrics C-V successfully!\n")
        return np.matmul(H_sum_by_C_to_V, map_H_line_to_row)
        # H_sum_by_C_to_V 是横向更新的矩阵，map_H_line_to_row 是将纵向edge转为横向edge的矩阵


class BP_NetDecoder:
    def __init__(self, H, batch_size):  # 校验矩阵，外部传入
        # 设置tf 初始化的模式
        self.initializer = tf.truncated_normal_initializer(mean=1, stddev=0.1)
        _, self.v_node_num = np.shape(H)  #  获取变量节点长度（即码元的长度 576）
        ii, jj = np.nonzero(H)  # 返回校验矩阵H中非零元素的索引，x轴依次返回给ii, y轴依次返回给jj，也就是说，非零元素的坐标表示位(ii,jj)
        loc_nzero_row = np.array([ii, jj])  # 将 ii 和 jj 组合起来了
        self.num_all_edges = np.size(loc_nzero_row[1, :])  # 获取非零元素的数量，同时也被称为edge，横坐标或者纵坐标的数量即edge数量
        gm1 = GetMatrixForBPNet(H[:, :], loc_nzero_row)  #
        self.H_sumC_to_V = gm1.get_Matrix_CV()  # 返回：C->V 的转换矩阵
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y = gm1.get_Matrix_VC()  # 返回：初始化的变量节点、V->C 的转换矩阵、输出层的转换矩阵
        # ---------改为 tf.Variable ----------
        self.H_x_to_xe0 = tf.Variable(self.H_x_to_xe0, dtype=tf.float32)
        # self.H_sumV_to_C = tf.Variable(self.H_sumV_to_C, dtype=tf.float32)
        self.H_xe_v_sumc_to_y = tf.Variable(self.H_xe_v_sumc_to_y, dtype=tf.float32)
        # ------------------------------------
        self.batch_size = batch_size
        self.llr_placeholder = tf.placeholder(tf.float32, [batch_size, self.v_node_num], name="llr_placeholder")
        # -----------新增变量------------
        self.x_bit_placeholder = tf.placeholder(tf.int8, [batch_size, self.v_node_num])
        self.train_bp_network = True
        # ---------- BP 网络的参数 -------------
        self.V_to_C_params = {}
        self.C_to_V_params = {}
        self.BP_layers = 2
        # ---------- 构建稀疏转换 --------------
        # --------- 将参数放到数组中 ----------
        i, j = np.nonzero(self.H_sumC_to_V)
        indices = []
        for idx in range(len(i)):
            tmp = [0, 0]
            tmp[0] = i[idx]
            tmp[1] = j[idx]
            indices.append(tmp)
        values = tf.Variable(np.ones(len(i)), dtype=tf.float32)

        for layer in range(self.BP_layers):
            self.C_to_V_params[layer] = tf.SparseTensor(indices=indices, values=tf.identity(values)
                                                        , dense_shape=self.H_sumC_to_V.shape)

        i, j = np.nonzero(self.H_sumV_to_C)
        indices = []
        for idx in range(len(i)):
            tmp = [0, 0]
            tmp[0] = i[idx]
            tmp[1] = j[idx]
            indices.append(tmp)
        values = tf.Variable(np.ones(len(i)), dtype=tf.float32)

        for layer in range(self.BP_layers):
            self.V_to_C_params[layer] = tf.SparseTensor(indices=indices, values=tf.identity(values)
                                                        , dense_shape=self.H_sumV_to_C.shape)

        # ---------改为 tf.Variable ----------
        # self.H_sumC_to_V = tf.Variable(self.H_sumC_to_V, dtype=tf.float32)
        # -----------------------------------
        # ------- 本来在最下面一行 -------------
        # init = tf.global_variables_initializer()
        # self.sess = tf.Session()  # open a session
        # print('Open a tf session!')
        # self.sess.run(init)
        # -------------------------------------
        # --------------------不带训练参数的BP译码网络------------------
        if not self.train_bp_network:
            self.llr_into_bp_net, self.xe_0, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out, self.sigmoid_out = self.build_network()
        # -----------------带训练参数的BP译码网络的参数矩阵--------------
        else:  # 之后考虑为每个 H_sumC_to_V 和 H_sumV_to_C 单独进行变量随机化
            # ii, jj = np.shape(self.H_sumC_to_V)
            # self.H_sum_param = tf.Variable(tf.random_normal([ii._value, jj._value], mean=1, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="H_param")
            # self.H_sumV_to_C = tf.multiply(self.H_sumV_to_C, self.H_sum_param)
            # self.H_sumC_to_V = tf.multiply(self.H_sumC_to_V, self.H_sum_param)
            self.llr_into_bp_net, self.xe_0, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out, self.sigmoid_out = self.build_trained_bp_network()
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder
        # self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_into_bp_net))  # transpose the llr matrix to adapt to the matrix operation in BP net decoder.

        self.sess = tf.Session()  # open a session
        # ---- tmp print --------------
        # print(self.sess.run((self.C_to_V_params[0])))
        print('Open a tf session!')
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def __del__(self):
        self.sess.close()
        print('Close a tf session!')


    def atanh(self, x):
        x1 = tf.add(1.0, x)
        x2 = tf.subtract((1.0), x)
        x3 = tf.divide(x1, x2)
        x4 = tf.log(x3)
        return tf.divide(x4, (2.0))

    def one_bp_iteration(self, xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, xe_0):
        """
        :param xe_v2c_pre_iter: (2040, 5000) ,xe_v2c_pre_iter 是上一轮的变量节点，即非初始化的变量节点
        :param H_sumC_to_V: (2040, 2040) 纵向排列调整为横向排列，同时横向更新
        :param H_sumV_to_C: (2040, 2040) 横向排列调整为纵向排列，同时纵向更新
        :param xe_0: (2040, 5000) 初始化的变量节点
        :return: 
        """
        xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
        xe_tanh = tf.to_float(xe_tanh)
        xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
        xe_sum_log_img = tf.matmul(H_sumC_to_V, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
        xe_sum_log_real = tf.matmul(H_sumC_to_V, tf.log(1e-8 + tf.abs(xe_tanh)))
        xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        xe_c_sumv = tf.add(xe_0, tf.matmul(H_sumV_to_C, xe_v_sumc))
        return xe_v_sumc, xe_c_sumv  # xe_v_sumc 是输出层，xe_c_sumv 是这一轮BP的输出，下一轮的输入

    def multiple_bp_iteration(self, xe_v2c_pre_iter, xe_0):
        """
        :param xe_v2c_pre_iter: (2040, 5000) ,xe_v2c_pre_iter 是上一轮的变量节点，即非初始化的变量节点
        :param H_sumC_to_V: (2040, 2040) 纵向排列调整为横向排列，同时横向更新
        :param H_sumV_to_C: (2040, 2040) 横向排列调整为纵向排列，同时纵向更新
        :param xe_0: (2040, 5000) 初始化的变量节点
        :return: 
        """
        # ------------- 创建可训练权重矩阵 -----------------------
        for layer in range(self.BP_layers):
            if 0 != layer:
                xe_v2c_pre_iter = xe_c_sumv

            xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
            xe_tanh = tf.to_float(xe_tanh)
            xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
            xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.C_to_V_params[layer], tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
            xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.C_to_V_params[layer], tf.log(1e-8 + tf.abs(xe_tanh)))
            xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
            xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
            xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
            xe_pd_modified = tf.add(xe_product, xe_product_temp)
            xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
            xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.V_to_C_params[layer], xe_v_sumc))
        # # ------------- 一轮 bp 迭代 -----------------------------
        # xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
        # xe_tanh = tf.to_float(xe_tanh)
        # xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
        # xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.w_C_to_V1, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
        # xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.w_C_to_V1, tf.log(1e-8 + tf.abs(xe_tanh)))
        # xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        # xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
        # xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        # xe_pd_modified = tf.add(xe_product, xe_product_temp)
        # xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        # xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.w_V_to_C1, xe_v_sumc))
        # # --------------- 二轮 dp 迭代 ---------------------------
        # xe_v2c_pre_iter = xe_c_sumv
        # xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
        # xe_tanh = tf.to_float(xe_tanh)
        # xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
        # xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.w_C_to_V2, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]),
        #                                                     [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
        # xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.w_C_to_V2, tf.log(1e-8 + tf.abs(xe_tanh)))
        # xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        # xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
        # xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        # xe_pd_modified = tf.add(xe_product, xe_product_temp)
        # xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        # xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.w_V_to_C2, xe_v_sumc))
        # # --------------- 三轮 dp 迭代 ---------------------------
        # xe_v2c_pre_iter = xe_c_sumv
        # xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
        # xe_tanh = tf.to_float(xe_tanh)
        # xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
        # xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.w_C_to_V3, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]),
        #                                                     [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
        # xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.w_C_to_V3, tf.log(1e-8 + tf.abs(xe_tanh)))
        # xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        # xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
        # xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        # xe_pd_modified = tf.add(xe_product, xe_product_temp)
        # xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        # xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.w_V_to_C3, xe_v_sumc))
        # # --------------- 四轮 dp 迭代 ---------------------------
        # xe_v2c_pre_iter = xe_c_sumv
        # xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
        # xe_tanh = tf.to_float(xe_tanh)
        # xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
        # xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.w_C_to_V4, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]),
        #                                                     [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
        # xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.w_C_to_V4, tf.log(1e-8 + tf.abs(xe_tanh)))
        # xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        # xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
        # xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        # xe_pd_modified = tf.add(xe_product, xe_product_temp)
        # xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        # xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.w_V_to_C4, xe_v_sumc))
        # # --------------- 五轮 dp 迭代 ---------------------------
        # xe_v2c_pre_iter = xe_c_sumv
        # xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))  # 除法 tanh(ve_v3c_pre_iter/2.0)
        # xe_tanh = tf.to_float(xe_tanh)
        # xe_tanh_temp = tf.sign(xe_tanh)  # 这一步的sign的作用，是将值重新变为-1，0，1这三种
        # xe_sum_log_img = tf.sparse_tensor_dense_matmul(self.w_C_to_V5, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]),
        #                                                     [3.1415926]))  # tf.multiply 矩阵按元素相乘, tf.matmul 则是标准的矩阵相乘
        # xe_sum_log_real = tf.sparse_tensor_dense_matmul(self.w_C_to_V5, tf.log(1e-8 + tf.abs(xe_tanh)))
        # xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        # xe_product = tf.real(tf.exp(xe_sum_log_complex))  # xe_sum_log_real
        # xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        # xe_pd_modified = tf.add(xe_product, xe_product_temp)
        # xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        # xe_c_sumv = tf.add(xe_0, tf.sparse_tensor_dense_matmul(self.w_V_to_C5, xe_v_sumc))

        return xe_v_sumc, xe_c_sumv  # xe_v_sumc 是输出层，xe_c_sumv 是这一轮BP的输出，下一轮的输入

    def build_network(self):  # build the network for one BP iteration
        # 还需要构建一段由 u_coded_bits 和 SNR 到 llr 的网络。
        # BP initialization
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32))  # 建立了一个矩阵变量（576 * 5000)，576 是码元，5000是每次5000个码元为一个batch
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net)  # 横向edge初始化(H_x_to_xe0:shape=(2040, 576), llr_into_bp_net:shape=(576, 5000) => (2040, 5000)
        xe_v2c_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32))  # the v->c messages of the previous iteration, shape=(2040, 5000)
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe_0)  # 将 xe_0 赋值给 ve_v2c_pre_iter_assign

        # one iteration
        H_sumC_to_V = tf.constant(self.H_sumC_to_V, dtype=tf.float32)  # shape=(2040, 2040)
        H_sumV_to_C = tf.constant(self.H_sumV_to_C, dtype=tf.float32)  # shape=(2040, 2040)
        # 上面两个变量改成下面的这两句
        # H_sumC_to_V = self.H_sumC_to_V
        # H_sumV_to_C = self.H_sumV_to_C
        # --------------------------
        xe_v_sumc, xe_c_sumv = self.one_bp_iteration(xe_v2c_pre_iter, H_sumC_to_V, H_sumV_to_C, xe_0)  # (2040, 5000), (2040, 2040), (2040, 2040), (2040, 5000)
        # xe_v_sumc 是纵向排列的edge，xe_c_sumv 是横向排列的edge
        # 横向排列的edge正好是每轮BP的输出，而纵向排列的BP则是可以作为输出层的前一个数据层
        # start the next iteration
        start_next_iteration = xe_v2c_pre_iter.assign(xe_c_sumv)

        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))  # H_xe_sumc_to_y 是输出层的转换矩阵，xe_v_sumc 是纵向排列的edge
        # sigmoid_out = tf.sigmoid(tf.transpose(bp_out_llr))
        sigmoid_out = tf.sigmoid(bp_out_llr)
        dec_out = tf.transpose(tf.floordiv(1-tf.to_int32(tf.sign(bp_out_llr)), 2), name="output_node_tanspose")

        return llr_into_bp_net, xe_0, xe_v2c_pre_iter_assign, start_next_iteration, dec_out, sigmoid_out

    def build_trained_bp_network(self):  # build the network for one BP iteration
        # 还需要构建一段由 u_coded_bits 和 SNR 到 llr 的网络。
        # BP initialization
        # llr_into_bp_net = tf.placeholder(dtype=tf.float32, shape=[self.v_node_num, self.batch_size], name="llr_into_bp_net_tensor")  # 整个BP网络的输入
        llr_into_bp_net = tf.Variable(np.ones([self.v_node_num, self.batch_size], dtype=np.float32))  # 建立了一个矩阵变量（576 * 5000)，576 是码元，5000是每次5000个码元为一个batch
        xe_0 = tf.matmul(self.H_x_to_xe0, llr_into_bp_net)  # 横向edge初始化(H_x_to_xe0:shape=(2040, 576), llr_into_bp_net:shape=(576, 5000) => (2040, 5000)
        xe_v2c_pre_iter = tf.Variable(np.ones([self.num_all_edges, self.batch_size], dtype=np.float32))  # the v->c messages of the previous iteration, shape=(2040, 5000)
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe_0)  # 将 xe_0 赋值给 ve_v2c_pre_iter_assign

        # one iteration
        # H_sumC_to_V = tf.constant(self.H_sumC_to_V, dtype=tf.float32)  # shape=(2040, 2040)
        # H_sumV_to_C = tf.constant(self.H_sumV_to_C, dtype=tf.float32)  # shape=(2040, 2040)
        # 上面两个变量改成下面的这两句，在不训练的情况下，变量和常量没有影响。
        # H_sumC_to_V = self.H_sumC_to_V
        # H_sumV_to_C = self.H_sumV_to_C
        # --------------------------
        xe_v_sumc, xe_c_sumv = self.multiple_bp_iteration(xe_v2c_pre_iter, xe_0)  # (2040, 5000), (2040, 2040), (2040, 2040), (2040, 5000)
        # xe_v_sumc 是纵向排列的edge，xe_c_sumv 是横向排列的edge
        # 横向排列的edge正好是每轮BP的输出，而纵向排列的BP则是可以作为输出层的前一个数据层
        # start the next iteration
        start_next_iteration = xe_v2c_pre_iter.assign(xe_c_sumv)

        # get the final marginal probability and decoded results
        bp_out_llr = tf.add(llr_into_bp_net, tf.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc))  # H_xe_sumc_to_y 是输出层的转换矩阵，xe_v_sumc 是纵向排列的edge
        # sigmoid_out = tf.sigmoid(tf.transpose(bp_out_llr))
        sigmoid_out = tf.sigmoid(bp_out_llr)
        dec_out = tf.transpose(tf.floordiv(1-tf.to_int32(tf.sign(bp_out_llr)), 2), name="output_node_tanspose")

        return llr_into_bp_net, xe_0, xe_v2c_pre_iter_assign, start_next_iteration, dec_out, sigmoid_out

    '''
    python -m tensorflow.python.tools.freeze_graph --input_checkpoint=model/bp_model/bp_model.ckpt --input_binary=false --output_graph=model/bp_model/frozen.pb --input_graph=model/bp_model/bp_model.pbtxt --output_node_names=output_node_tanspose
    '''
    # 可训练的BP decode
    def decode(self, llr_in, bp_iter_num):
        real_batch_size, num_v_node = np.shape(llr_in)  # llr_in 就是BP译码的初始化
        if real_batch_size != self.batch_size:  # padding zeros
            llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
            # original llr array.
        self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})  # llr应该只是数据层

        # 每个作用域内 tensor 都需要初始化一下
        # init = tf.global_variables_initializer()
        # self.sess.run(init)

        # 尝试保存网络
        # saver = tf.train.Saver()
        # save_dir = "model/bp_model/"

        self.sess.run(self.xe_v2c_pre_iter_assign)  #
        # for iter in range(0, bp_iter_num-1):
        #     self.sess.run(self.start_next_iteration)  # run start_next_iteration时表示当前一轮BP的输出
        y_dec = self.sess.run(self.dec_out)  # dec_out 则是最终输出层
        # sigmoid_out = self.sess.run(self.sigmoid_out)
        # saver.save(self.sess, save_dir + "bp_model.ckpt")

        if real_batch_size != self.batch_size:
            y_dec = y_dec[0:real_batch_size, :]

        return y_dec

    # 不可训练的BPdecode
    # def decode(self, llr_in, bp_iter_num):
    #     real_batch_size, num_v_node = np.shape(llr_in)
    #     if real_batch_size != self.batch_size:  # padding zeros
    #         llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
    #         # original llr array.
    #     self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})
    #     self.sess.run(self.xe_v2c_pre_iter_assign)
    #     for iter in range(0, bp_iter_num-1):
    #         self.sess.run(self.start_next_iteration)
    #     y_dec = self.sess.run(self.dec_out)
    #     if real_batch_size != self.batch_size:
    #         y_dec = y_dec[0:real_batch_size, :]
    #
    #     return y_dec

    def train_decode_network(self, bp_iter_num, SNRset, batch_size, ch_noise_normalize, linear_code):
        """
        神经网络的：
        输入 u_coded_bits 和 SNR
        输出 LLR
        :param llr_in: 这里 LLR 是和 u_coded_bits 相反的 
        :param bp_iter_num: 
        :param u_coded_bits: 处理方法是将 u_coded_bits 中的 0 和 1 翻过来 
        :return: 
        """
        noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file)

        # 尝试保存网络
        saver = tf.train.Saver()
        save_dir = "model/bp_model/"
        G_matrix = linear_code.G_matrix  # 用于产生 u_coded_bits 样本的生成矩阵
        rng = np.random.RandomState(0)  # 随机数对象
        # u_coded_bits = 1 - u_coded_bits  # 翻转 u_coded_bits 以和 llr_in 相匹配
        # u_coded_bits = tf.Variable(u_coded_bits, dtype=tf.float32, name="u_coded_bits")

        # 先根据 SNR 和 u_coded_bits 求出 llr_in，这样子就可以使用下面的代码了
        _, N = G_matrix.shape  # 传入一个u_coded_bits，用于分析其shape
        u_coded_bits_tensor = tf.placeholder(dtype=tf.float32, shape=[batch_size, N], name="u_coded_bits_tensor")  # 整个BP网络的输入
        SNR_tensor = tf.placeholder(dtype=tf.float32, shape=[1], name="SNR_tensor")  # 整个BP网络的输入
        s_mod_tensor = tf.add(tf.multiply(u_coded_bits_tensor, -2), 1)  # BPSK 调制
        ch_noise_sigma_tensor = tf.square(
            tf.truediv(tf.truediv([1.0], tf.pow(10.0, tf.truediv(SNR_tensor, [10.0]))), [2.0]))  # 计算噪声的标准差
        ch_noise_normalize_tensor = tf.constant(ch_noise_normalize, dtype=tf.float32)  # 噪声矩阵
        ch_noise_tensor = tf.multiply(ch_noise_normalize_tensor, ch_noise_sigma_tensor)  # 生成真正的噪声
        y_receive_tensor = tf.add(s_mod_tensor, ch_noise_tensor)  # 模拟高斯白噪声下的接收信号
        LLR_tensor = tf.truediv(tf.multiply(y_receive_tensor, 2.0),
                                tf.multiply(ch_noise_sigma_tensor, ch_noise_sigma_tensor), name="LLR_tensor")  # 计算接收信号的对数似然比
        # ------------------------------------------------------------------
        # llr_in = self.sess.run(LLR_tensor, feed_dict={SNR_tensor: [SNR], u_coded_bits_tensor: u_coded_bits})
        # 每个作用域内 tensor 都需要初始化一下，但是一旦这样子初始化后，就会将之前训练的结果全部清楚。。。，所以最好放到一个
        # real_batch_size, num_v_node = np.shape(llr_in)  # llr_in 就是BP译码的初始化
        # if real_batch_size != self.batch_size:  # padding zeros，目前还没进入这个选项，先忽略，之后改成tensor模式
        #     llr_in = np.append(llr_in, np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32), 0)  # re-create an array and will not influence the value in
            # original llr array. 将 np.zeros 这部分放到llr_in 后面几行（补零操作），只改变行，不改变列。
            # 可以改成下面的 tensor 形式
            # llr_in_tensor = tf.concat([LLR_tensor, tf.Variable(np.zeros([self.batch_size-real_batch_size, num_v_node], dtype=np.float32))], axis=0)
        # self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})  # llr应该只是数据层

        # self.sess.run(self.xe_v2c_pre_iter_assign)  #
        # for iter in range(0, bp_iter_num-1):  # 根据BP迭代的次数，搭建BP网络
        #     self.sess.run(self.start_next_iteration)  # run start_next_iteration时表示当前一轮BP的输出
        # y_dec = self.sess.run(self.dec_out)  # dec_out 则是最终输出层
        # sigmoid_out = self.sess.run(self.sigmoid_out) #

        # cross_entropy = -tf.reduce_sum(u_coded_bits_tensor * tf.log(self.sigmoid_out))  #　* 是按元素相乘，u_coded_bits=(5000,6);sigmoid_out=(6,5000)
        # cross_entropy = -tf.reduce_sum(tf.transpose(self.llr_into_bp_net) * tf.log(self.sigmoid_out))  # * 是按元素相乘，u_coded_bits=(5000,6);sigmoid_out=(6,5000)
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        tf.train.start_queue_runners(sess=self.sess)
        # # 每个作用域内 tensor 都需要初始化一下，但是一旦这样子初始化后，就会将之前训练的结果全部清楚。。。，所以最好放到一个
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        # ------------------ 如果调制 init 的位置 --------？？？？ -----------


        cross_entropy = -tf.reduce_sum(self.llr_into_bp_net * tf.log(self.sigmoid_out))  # * 是按元素相乘，u_coded_bits=(5000,6);sigmoid_out=(6,5000)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        # ------------- 恢复保存的神经网络 ------------------
        # if os.path.exists(save_dir + "bp_model.ckpt.meta"):
        #     saver.restore(self.sess, save_dir + "bp_model.ckpt")
        # --------------------------------------------------
        tmp = 0
        for SNR in SNRset:
            real_batch_size = batch_size
            for i in range(1):  # 20000
                # 需要一个更新输入数据的过程
                # u_coded_bits = self.generate_inputs(G_matrix, batch_size, rng)  # 使用自带的数据生成函数
                x_bits, _, s_mod, channel_noise, y_receive, LLR = lbc.encode_and_transmission(G_matrix, SNR, real_batch_size, noise_io)
                # llr_in = self.sess.run(LLR_tensor, feed_dict={SNR_tensor: [SNR], u_coded_bits_tensor: u_coded_bits})
                # self.llr_assign = self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: llr_in})  # llr应该只是数据层
                # train_step.run(feed_dict={SNR_tensor: [SNR], u_coded_bits_tensor: u_coded_bits}, session=self.sess)
                # self.sess.run(train_step, feed_dict={SNR_tensor: [SNR], u_coded_bits_tensor: u_coded_bits})   # 重新修改网络的输入为 llr_in
                # --------------------------------------------------------------------------------------------
                self.sess.run(self.llr_assign, feed_dict={self.llr_placeholder: LLR})
                # self.sess.run(self.xe_v2c_pre_iter_assign)  #
                # for iter in range(0, bp_iter_num - 1):  # 根据BP迭代的次数，搭建BP网络
                #     self.sess.run(self.start_next_iteration)  # run start_next_iteration时表示当前一轮BP的输出
                # --------------------------------------------------------------------------------------------
                tmp = LLR
                self.sess.run(train_step)  # 重新修改网络的输入为 llr_in
        # ------------- 保存训练好的神经网络 -----------------
        saver.save(self.sess, save_dir + "bp_model.ckpt")
        print(self.sess.run(tf.sparse.to_dense(self.V_to_C_params[0])))
        # ---------------------------------------------------
        # if real_batch_size != self.batch_size:
        #     y_dec = y_dec[0:real_batch_size, :]
        #
        # return y_dec

    # def restore_bp_network(self, sess):
    #     # restore some layers
    #     save_dict = {}
    #     for layer in range(restore_layers_in):
    #         save_dict[self.conv_filter_name[layer]] = self.conv_filter[layer]
    #         save_dict[self.bias_name[layer]] = self.bias[layer]
    #     model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
    #     model_id_str = model_id_str[1:(len(model_id_str)-1)]
    #     model_folder = format("%snetid%d_model%s" % (self.net_config.model_folder, self.net_id, model_id_str))
    #     restore_model_name = format("%s/model.ckpt" % model_folder)
    #     saver_restore = tf.train.Saver(save_dict)
    #     saver_restore.restore(sess_in, restore_model_name)
    #     print("Restore the first %d layers.\n" % restore_layers_in)

    def generate_inputs(self, G_matrix, batch_size, rng):
        K, N = np.shape(G_matrix)
        if rng == 0:
            x_bits = np.random.randint(0, 2, size=(batch_size, K))
        else:
            x_bits = rng.randint(0, 2, size=(batch_size, K))  # 随机数种子rng用于生产随机的输入码字
        # coding
        u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # G_matrix
        return u_coded_bits
 # [[{{node beta2_power/read}}]]