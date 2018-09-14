import cv2
import os
import numpy as np
import random
import tensorflow  as tf
path = r'D:\_____\Works\PyProject\tensortest\image_cv\data'
# img=cv2.imread(path+r'\train\1.png')
# img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img=img/255
# print(img)
# l = os.listdir(r'./test')
# print(l[:20])
#
# l=np.sort(l)
# print(l[:20])
# i=0
# img = cv2.imread(r'./test/' + str(i) + r'.png')
# print(img)
# img=cv2.imread(r'./test/'+l[1])
# print(img)
def load_data_label(lenth,path1,path2):
    data_list=[]
    for i in range(lenth):
        img=cv2.imread(path1+str(i)+r'.png')
        # img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=img/255
        data_list.append(img)
    data_arr=np.array(data_list)
    # print(data_arr.shape)
    with open(path2,'r') as f:
        label_list=str(f.read())
    label_list=label_list.split(',')
    label_list=list(map(int,label_list))
    labels=[]
    for i in label_list:
        label_i=[0,0,0,0,0,0,0,0,0,0]
        label_i[i]=1
        labels.append(label_i)
        # print(label_list)
    # print(labels[:20])
    return data_list,labels[:lenth]

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
xs = tf.placeholder(tf.float32, [None, 28,28,3]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,3])
print(x_image.shape)

# conv1 layer ##
W_conv1=weight_variable([5,5,3,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


## func1 layer ##
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
## func2 layer ##

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# print(ys.shape,prediction.shape)
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply( ys , tf.log(prediction)),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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


def train_next_batch(num,data,label):
    next_batct_data=[]
    next_batch_label=[]
    value=range(0,1000)
    batch=random.sample(value,num)
    # print(batch)
    for i in batch:
        next_batct_data.append(data[i])
        next_batch_label.append(label[i])
    next_batch_label= np.array(next_batch_label).reshape(-1,10)
    return next_batct_data,next_batch_label
train_path1,train_path2=path+r'/train/',path+r'/train_label.txt'
test_path1,test_path2=path+r'/test/',path+r'/test_label.txt'

train_img,train_label=load_data_label(1000,train_path1,train_path2)
test_img,test_label=load_data_label(1000,test_path1,test_path2)
# img1,lab=train_next_batch(100,train_img,train_label)

for i in range(1000):
    batch_xs,batch_ys=train_next_batch(100,train_img,train_label)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i%200==0:
        test_xs,test_ys=test_img[:100],test_label[:100]
        print(compute_accuracy(test_xs,test_ys))
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
