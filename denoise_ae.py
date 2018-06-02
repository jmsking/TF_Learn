#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

'''
去噪Auto-Encoder(单隐藏层)
'''

class AdditiveGaussianNoiseAutoEncoder():
  '''
  加性高斯噪声自编码器
  '''
  def __init__(self, n_input, n_hidden, scale = 0.01, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(0.001)):
    '''
    初始化
    Params:
      n_input: 网络输入节点数
      n_hidden: 网络隐藏层节点数
      scale: 噪声
      optimizer: 网络优化器
    '''
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.scale = scale
    self.optimizer = optimizer
    self.transfer_function = transfer_function
    self.x = tf.placeholder(tf.float32, [None, n_input])
    self.weights = self._initialize_weights()
    
  def xavier_init(self, constant = 1):
    '''
    xavier初始化器
    Params:
      n_in: 输入层节点数
      n_out: 隐藏层节点数
      constant: 控制阈值范围宽度
    '''
    low = -constant * tf.sqrt(6.0 / (self.n_input + self.n_hidden))
    high = constant * tf.sqrt(6.0 / (self.n_input + self.n_hidden))
    return tf.random_uniform((self.n_input, self.n_hidden), minval = low, maxval = high, dtype = tf.float32)
    
  def standard_scaler(self, x_train, x_test):
    '''
    对数据进行归一化(均值为0, 方差为1)
    Params:
      x_train: 待归一化的训练集
      x_test: 待归一化的测试集
    Returns:
      归一化后的数据
    '''
    standard = prep.StandardScaler().fit(x_train)
    x_train = standard.transform(x_train)
    x_test = standard.transform(x_test)
    return x_train, x_test
    
  def obtain_mini_batch(self, x_train, n_samples, batch_size):
    '''
    获取Mini-Batch训练样本
    Params:
      x_train: 训练样本
    n_samples: 训练样本数
      batch_size: 批次大小
    Returns:
      batch_xs: 返回mini-batch训练样本
    '''
    batch = n_samples//batch_size
    for i in range(batch):
      start = i * batch_size
      batch_xs = x_train[start:start+batch_size,:]
      yield batch_xs
    
  def _initialize_weights(self):
    net_weights = {}
    net_weights['W1'] = tf.Variable(self.xavier_init())
    net_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
    net_weights['W2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32)
    net_weights['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)
    return net_weights
    
  def build_net(self):
    '''
    构建AutoEncoder网络结构
    '''
    W1 = self.weights['W1']
    b1 = self.weights['b1']
    W2 = self.weights['W2']
    b2 = self.weights['b2']
    self.compress = self.transfer_function(tf.matmul(self.x+self.scale*tf.random_normal((self.n_input,)), W1) + b1)
    self.reconstruct = tf.matmul(self.compress, W2) + b2
    
  def build_loss(self):
    '''
    构建Loss函数
    '''
    self.loss = 0.5*tf.reduce_sum(tf.pow(self.reconstruct - self.x, 2))
    
  def start_train(self):
    '''
    开始训练
    '''
    epoches = 50
    batch_size = 128
    display_iter = 1
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    x_train = mnist.train.images
    x_test = mnist.test.images
    n_samples = mnist.train.num_examples
    print(n_samples)
    x_train, x_test = self.standard_scaler(x_train, x_test)
    print(x_train.shape)
    print(x_test.shape)
    self.build_net()
    self.build_loss()
    train_step = self.optimizer.minimize(self.loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      for epoch in range(epoches):
        avg_cost = 0
        for bat, batch_xs in enumerate(self.obtain_mini_batch(x_train, n_samples, batch_size)):
          _,cost = sess.run([train_step, self.loss], feed_dict={self.x:batch_xs})
          avg_cost += cost / n_samples * batch_size		  
        if epoch % display_iter == 0:
          print("Epoches: {:>3d}/{} - Loss: {:>4.2f}".format(epoch+1, epoches, avg_cost))
          
      cost = sess.run(self.loss, feed_dict={self.x:x_test})
      print("Test Cost: {:>4.2f}".format(cost))

n_hidden = 200      
auto_encoder = AdditiveGaussianNoiseAutoEncoder(28*28, n_hidden)
auto_encoder.start_train()