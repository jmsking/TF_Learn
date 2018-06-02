#! /usr/bin/python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Program 1：识别MNIST手写字体
过程：
  (1)：加载MNIST数据
  (2)：定义变量
  (3)：定义算法公式
  (4)：定义Loss函数
  (5)：定义优化算法
  (6)：迭代训练
  (7)：模型评估
'''

def mnist_recog():
  '''
  加载MNIST数据
  '''
  mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
  print(mnist.train.images.shape, mnist.train.labels.shape)
  print(mnist.test.images.shape, mnist.test.labels.shape)
  print(mnist.validation.images.shape, mnist.validation.labels.shape)
  
  '''
  定义变量
  '''
  x = tf.placeholder(tf.float32, [None, 28*28])
  y = tf.placeholder(tf.float32, [None, 10])
  w = tf.Variable(tf.zeros([28*28, 10]))
  b = tf.Variable(tf.zeros([10]))
  
  '''
  定义算法公式
  softmax分类
  '''
  y_ = tf.nn.softmax(tf.matmul(x,w)+b)
  
  '''
  定义Loss函数
  交叉熵 cross_entropy
  '''
  cross_entropy = -tf.reduce_sum(y * tf.log(y_))
  
  '''
  定义优化算法
  GradientDescent随机梯度下降算法SGD
  '''
  learn_rate = 0.005
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
  
  '''
  迭代训练
  '''
  display_iter = 50
  epoches = 1000
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(epoches):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      _,loss = sess.run([train_step, cross_entropy], feed_dict={x:batch_xs, y:batch_ys})
      if epoch % display_iter == 0:
        print('Epoch: {:>3}/{} - Training loss: {:>6.3f}'
            .format(epoch+1, epoches, loss))
    
    '''
    模型评估
    Accuracy准确率判定
    '''
    correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
    print(sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels}))
    print(accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    print(sess.run(accuracy, feed_dict={x:mnist.validation.images, y:mnist.validation.labels}))

mnist_recog()