import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#这是我一个练习的例子，做到一半实在不会做了，
# 原型是根据莫凡老师的rnn回归例子，
# 希望朋友能帮忙完善一下，
# 思路如下，读取a（a是）列数据分割成1900行，再做点阵相乘，乘出来后变成l1，l2在对c列进行同样的操作，然后和l1相加。
#最后求出整体最低的cost，可视化不会做……怎么保存训练结果和读取也不会,方面就顺手帮忙做了 不方便就我自己学了做了，


df = pd.read_excel("./gbtc.xlsx",
filepath_or_buffer="./gbtc.xlsx",
sep=",",
error_bad_lines=False,
na_values="NULL",
usecols=[0,1,2,3,4,5,6,7,8,9,10,11],
nrows=1900,
lineterminator="\n")
diminput = 1
dimhidden = 1900 #隐层128个
output_size = 1
cell_size = 1900
TIME_STEPS =1
nsteps = 1   #
batch_size = df.iloc[:, 1].size
a = tf.constant((df["百分比变化(change_rate)"]))#  chang the dtype
a = tf.to_float(a)
a = tf.reshape(a, (batch_size, 1))
b = tf.constant((df["时间"]))
b = tf.to_float(b)
c = tf.constant((df["MV2"]))
c = tf.to_float(c)
ys = tf.constant(df["结果"])
ys = tf.to_float(ys)
#定义需要的变量
x = tf.placeholder(tf.float32, [None, nsteps, diminput])#这里没有dict 已经混乱不知道如何赋值了…………
#x = [batch_size,nsteps，input_size]
weights = {"hidden": tf.Variable(tf.random_normal([batch_size]), dtype=tf.float32),
         'out': tf.Variable(tf.random_normal([batch_size, nsteps]), dtype=tf.float32)
           }
biases = {'out': tf.Variable(tf.random_normal([batch_size,nsteps]), dtype=tf.float32),
          "hidden": tf.Variable(tf.random_normal([batch_size]), dtype=tf.float32)}
class LSTMRNN():
    def addlayer(self,input,diminput,activation_function=None):
        hsplit = tf.split(input, num_or_size_splits=(batch_size))
        output = tf.multiply(hsplit, weights["hidden"]) +biases["hidden"]
        print(tf.shape(output))
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size, forget_bias=1.0)
        with tf.name_scope('initial_state'):
            cell_init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            cell_outputs, cell_final_state = tf.nn.dynamic_rnn(lstm_cell,inputs=x, initial_state=cell_init_state, time_major=False, dtype=tf.int32)   # 输入的三维数据[batch_size,num_steps, state_size]
        cell_outputs = tf.to_float(cell_outputs)  # haimei shiyong
        print(cell_outputs)
        with tf.name_scope('Wx_plus_b'):
            output = tf.multiply(cell_outputs, weights["out"]) + biases["out"]
        if self.activation_function is None:  #这里根据网上的案例是说会自动识别是否为线性关系，不是则自动调用激活函数
            output1 = output
        else:
            output1 = self.activation_function(output)
            return output1
    l1 = addlayer(a, 1, activation_function=tf.sigmoid)
    l2 = addlayer(c, 1, activation_function=tf.sigmoid) +l1#这里要对变量名重复利用，但是不太会弄
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(

            [tf.reshape(self.output1, [-1], name='reshape_pred')],

            [tf.reshape(self.ys, [-1], name='reshape_target')],

            [tf.ones([self.batch_size * self.nsteps], dtype=tf.float32)],

            average_across_timesteps=True,

            softmax_loss_function=self.ms_error,

            name='losses'

        )

        with tf.name_scope('average_cost'):
            self.cost = tf.div(

                tf.reduce_sum(losses, name='losses_sum'),

                batch_size,

              name='average_cost')

            tf.summary.scalar('cost', self.cost)
    @staticmethod

    def ms_error(labels, logits):

        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):

        initializer = tf.random_normal_initializer(mean=0., stddev=1., )

        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):

        initializer = tf.constant_initializer(0.1)

        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':

   # model = LSTMRNN(TIME_STEPS, diminput, output_size, cell_size, batch_size)

    sess = tf.Session()

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("logs", sess.graph)

    # tf.initialize_all_variables() no long valid from

    # 2017-03-02 if using tensorflow >= 0.12

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:

        init = tf.initialize_all_variables()

    else:

        init = tf.global_variables_initializer()

    sess.run(init)

