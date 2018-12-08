import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
input_size=28
input_step=28
hide_unit=100
out_class=10

lr=0.003
train_max=10000
batch_size=100

x=tf.placeholder(tf.float32,[None,input_step,input_size])
y=tf.placeholder(tf.float32,[None,out_class])

weight={
    'in': tf.Variable(tf.random_normal([input_size,hide_unit])),
    'out' :tf.Variable(tf.random_normal([hide_unit,out_class]))
}
bias={
    'in': tf.Variable(tf.random_normal([hide_unit,])),
    'out': tf.Variable(tf.random_normal([out_class,]))
}

def RNN(X,weight,bias):
    X=tf.reshape(X,[-1,input_size])

    X_in=tf.matmul(X,weight['in'])+bias['in']
    X_in=tf.reshape(X_in,[-1,input_step,hide_unit])
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(hide_unit,forget_bias=1.0,state_is_tuple=True)
    init_cell=lstm_cell.zero_state(batch_size,dtype=tf.float32)

    output,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_cell,time_major=False)
    print(final_state[1])
    result=tf.matmul(final_state[1],weight['out'])+bias['out']
    return  result

pred=RNN(x,weight,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*batch_size<train_max:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,input_step,input_size])
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        if step%200==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1