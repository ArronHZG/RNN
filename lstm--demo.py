import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import os, sys
#定义超参数 Hyperparameters
class Hp:
    batch_start = 0
    time_steps = 2
    batch_size = 20
    input_size = 7
    output_size = 1
    cell_size = 20
    learning_rate = 0.001
    layer_num=1
    ckpt_dir=os.path.basename(sys.argv[0]).split(".")[0]+"/ckpt/"
    log_dir=os.path.basename(sys.argv[0]).split(".")[0]+"/logs/"
#定义读取excel行数
NROWS=1938
#读取数据
def prepareData():
    df = pd.read_excel("./gbtc.xlsx",
                       filepath_or_buffer="./gbtc.xlsx",
                       sep=",",
                       error_bad_lines=False,
                       na_values="NULL",
                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],#不采用data列
                       nrows=NROWS,
                       lineterminator="\n")


    df=df.sort_values("时间")#按照时间戳升序排序
    seq=np.array(df.values,dtype="float32")#获得所有列并转为array
    seq=np.delete(seq,6,axis=1)#删除百分比变化(change_rate)
    seq=np.delete(seq,6,axis=1)#删除时间
    result=np.array(df["百分比变化(change_rate)"]).reshape([-1,1])
    xs=np.array(df["时间"]).reshape([-1,1])
    # xs-=1448035200
    # xs//=10000
    print(f"seq.shape: {seq.shape}")
    print(f"result.shape: {result.shape}")
    print(f"xs.shape: {xs.shape}")
    # 标准化数据,零均值单位方差
    standardized_seq = preprocessing.scale(seq)
    standardized_result=preprocessing.scale(result)
    # 标准化数据,将时间缩放到区间 [0, 1]
    standardized_xs=preprocessing.maxabs_scale(xs)
    return standardized_seq,standardized_result,standardized_xs
    # seq.shape: (1000, 7)
    # result.shape: (1000, 1)
    # xs.shape: (1000, 1)

def get_batch(standardized_seq,standardized_result,standardized_xs):
    xs = standardized_xs[Hp.batch_start:Hp.batch_start+Hp.time_steps*Hp.batch_size,:].reshape((Hp.batch_size, Hp.time_steps))

    #将数据转为若干最小批
    batch_seq=standardized_seq[Hp.batch_start:Hp.batch_start+Hp.time_steps*Hp.batch_size,:].reshape([-1,Hp.time_steps,Hp.input_size])
    batch_result=standardized_result[Hp.batch_start:Hp.batch_start+Hp.time_steps*Hp.batch_size,:].reshape([-1,Hp.time_steps,Hp.output_size])
    # 将时间转为若干最小批
    Hp.batch_start += Hp.time_steps
    if Hp.batch_start+Hp.time_steps*Hp.batch_size>=NROWS-1:
        Hp.batch_start=0
    return batch_seq, batch_result, xs
    # batch_seq.shape: (50, 20, 7)
    # batch_result.shape: (50, 20, 1)
    # batch_xs.shape: (50, 20)

class LSTMRNN():
    def __init__(self, Hp=Hp):
        self.n_steps = Hp.time_steps
        self.input_size = Hp.input_size
        self.output_size = Hp.output_size
        self.cell_size = Hp.cell_size
        self.batch_size = Hp.batch_size
        self.learning_rate=Hp.learning_rate
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, self.n_steps, self.output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def add_input_layer(self):

        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        #定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        #添加 dropout layer, 正则化方法，可以有效防止过拟合
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell)
        #调用 MultiRNNCell 来实现多层 LSTM
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * Hp.layer_num, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            #用全零来初始化state
            self.cell_init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.nn.relu(tf.matmul(l_out_x, Ws_out) + bs_out)

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == '__main__':
    standardized_seq, standardized_result, standardized_xs = prepareData()
    model = LSTMRNN()
    saver = tf.train.Saver(max_to_keep=1)
    min_cost = 10000000

    with tf.Session() as sess:
        try:
            model_file = tf.train.latest_checkpoint(Hp.ckpt_dir)
            saver.restore(sess, model_file)
            print("加载训练权重")
        except:
            print("没有训练权重")
        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(Hp.log_dir, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        plt.ion()
        plt.show()
        state = None
        for i in range(20000):
            seq, res, xs = get_batch(standardized_seq, standardized_result, standardized_xs)
            # print(f"xs.shape{xs.shape}")
            if i == 0:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state  # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)

            # plotting
            plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:Hp.time_steps], 'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.3)

            if i % 20 == 0:
                print(f'i:{i}    cost: ', round(cost, 4))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)
                plt.clf()#清屏
            if cost < min_cost:
                min_cost = cost
                saver.save(sess,Hp.ckpt_dir, global_step=i + 1)
