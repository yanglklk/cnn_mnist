# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:45:58 2018

@author: yanglklk
"""
import cv2
import os
import numpy as np
import random
import tensorflow  as tf
path = r'D:\works\PyProject\img\data'
img=cv2.imread(path+r'\test\300.jpg')
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img=img/255

xs = tf.placeholder(tf.float32, [None, 256,384,1]) # 28x28
ys = tf.placeholder(tf.float32, [None, 5])
keep_prob = tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,256,384,1])


def load_data_label(path1,path2):
    l=os.listdir(path1)
    data_list=[]
    for i in range(len(l)):
        img=cv2.imread(path1+l[i])
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=img/255
        img=img.reshape(256,384,1)
        data_list.append(img)
    data_arr=np.asarray(data_list)
    with open(path2,'r') as f:
        label_list=str(f.read())
    label_list=label_list.split(',')
    label_list=list(map(int,label_list))
    labels=[]
    for i in label_list:
        label_i=[0,0,0,0,0]
        label_i[i]=1
        labels.append(label_i)
        # print(label_list)
    # print(labels[:20])
    labels=np.asarray(labels,dtype=np.float32)
    return data_arr,labels

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network


# conv1 layer ##
W_conv1=weight_variable([5,5,1,5])
b_conv1=bias_variable([5])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2=weight_variable([5,5,5,10])
b_conv2=bias_variable([10])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


## func1 layer ##
W_fc1=weight_variable([64*96*10,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,64*96*10])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
## func2 layer ##

W_fc2=weight_variable([1024,5])
b_fc2=bias_variable([5])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# print(ys.shape,prediction.shape)
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(ys, tf.log(prediction)),
                                              reduction_indices=[1]))       # loss
loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)+(1-ys)*tf.log(1-prediction)))
train_step1= tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step2= tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)
train_step3= tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
# init=tf.initialize_all_variables()
sess.run(init)


def train_next_batch(num,data,label,range1):
    next_batch_data=[]
    next_batch_label=[]
    value=range(0,range1)
    batch=random.sample(value,num)
    #要再建list太慢了
    # for i in batch:
    #     next_batch_data.append(data[i])
    #     next_batch_label.append(label[i])
    next_batch_data=data[batch,:,:,:]
    next_batch_label=label[batch,:]
    return next_batch_data,next_batch_label
train_path1,train_path2=path+r'/train/',path+r'/train_label.txt'
test_path1,test_path2=path+r'/test/',path+r'/test_label.txt'
#
train_img,train_label=load_data_label(train_path1,train_path2)
test_img,test_label=load_data_label(test_path1,test_path2)
print(train_label.shape)
test=test_img[[2,22,45,64,82],:,:,:]

for i in range(5000):
    img, label = train_next_batch(50, train_img, train_label, 400)
    sess.run(train_step1, feed_dict={xs: img, ys: label, keep_prob: 0.5})
    if i%100==0:
        print(i)
        test_xs,test_ys=test_img,test_label
        print('acc',compute_accuracy(test_xs,test_ys))
for i in range(5000):
    img, label = train_next_batch(50, train_img, train_label, 400)
    sess.run(train_step2, feed_dict={xs: img, ys: label, keep_prob: 0.5})
    if i%100==0:
        print(i)
        test_xs,test_ys=test_img,test_label
        print('acc',compute_accuracy(test_xs,test_ys))
for i in range(5000):
    img, label = train_next_batch(50, train_img, train_label, 400)
    sess.run(train_step3, feed_dict={xs: img, ys: label, keep_prob: 0.5})
    if i%100==0:
        print(i)
        test_xs,test_ys=test_img,test_label
        print('acc',compute_accuracy(test_xs,test_ys))
# sess.run(prediction,feed_dict={xs:test,keep_prob:1})




















# test_img,test_label=np.array(test_img),np.array(test_label)
# img1,lab=train_next_batch(100,train_img,train_label)
# print(train_img.shape,test_img.shape)
# for i in range(100):
#     # batch_xs,batch_ys=train_next_batch(10,train_img,train_label,400)
#     # print(batch_xs.shape,batch_ys.shape)
#     sess.run(train_step, feed_dict={xs: test_img, ys: test_label, keep_prob: 0.5})
#     if i%20==0:
#         test_xs,test_ys=test_img,test_label
#         print(compute_accuracy(test_xs,test_ys))
# print(batch,lab[:5])
# for i in range(5):
#     cv2.imshow('1',img1[i])
#     cv2.waitKey(2000)
# batch1,batch2=train_next_batch(100,data,label)
# print(batch1.shape,batch2.shape)
# for i in range(1000):
#     batch_xs, batch_ys = data[:100],label[:100]
#         # train_next_batch(100,data,label)
#     # print(batch_xs[i].shape, batch_ys)
#     # batch_xs,batch_ys=np.array(batch_xs),np.array(batch_ys)
#     # print(batch_xs.shape,batch_ys.shape)
#     sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
#     if i%200==0:
#         test_xs,test_ys=data[:10],label[:10]
#         print(compute_accuracy(test_xs,test_ys))
# for i in range(5):
#     cv2.imshow('1',data[i])
#     cv2.waitKey(1000)
#     print(label[i])

