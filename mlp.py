#! /usr/bin/python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class MultiLayerPerceptron:

  '''
  构建多层感知器
  采用Dropout, Adagrad, ReLU
  '''
  
  def __init__(self, n_input=28*28, n_hidden=300, n_output=10, batch_size=128, 
    transfer_func = tf.nn.relu, optimizer = tf.train.AdagradOptimizer(0.03)):
    '''
    MLP初始化
    '''
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.n_output = n_output
    self.batch_size = batch_size
    self.transfer_func = transfer_func
    self.optimizer = optimizer
  
  def build_input(self):
    '''
    构建输入
    '''
    self.x = tf.placeholder(tf.float32, [None, self.n_input])
    self.y = tf.placeholder(tf.float32, [None, self.n_output])
    self.keep_prob = tf.placeholder(tf.float32)
    self.w1 = tf.Variable(tf.truncated_normal([self.n_input, self.n_hidden], stddev=0.1))
    self.b1 = tf.Variable(tf.zeros([self.n_hidden]))
    self.w2 = tf.Variable(tf.zeros([self.n_hidden, self.n_output]))
    self.b2 = tf.Variable(tf.zeros([self.n_output]))
    
  def build_net(self):
    '''
    构建网络结构,构建单隐藏层
    Returns:
      o_out: 网络输出
      loss: 损失函数
      train_step: 训练步骤
    '''
    # 构建隐藏层输出
    h_out = self.transfer_func(tf.matmul(self.x,self.w1)+self.b1)
    # Dropout
    h_out_dropout = tf.nn.dropout(h_out, self.keep_prob)
    # 构建输出层输出
    o_out = tf.nn.softmax(tf.matmul(h_out,self.w2)+self.b2)
    # 定义损失函数-交叉熵
    loss = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(o_out)))
    train_step = self.optimizer.minimize(loss)
    return o_out, loss, train_step
  
  def obtain_mini_batch(self, n_samples, train_x, train_y):
    '''
    获取Mini-Batch训练样本
    Params:
      n_samples: 训练样本数
      train_x: 训练样本集
      train_y: 训练样本标记
    Returns:
      batch_x, batch_y: mini-batch训练集
    '''
    n_batches = n_samples // self.batch_size
    for i in range(n_batches):
      start = i*self.batch_size
      batch_x = train_x[start:start+self.batch_size,:]
      batch_y = train_y[start:start+self.batch_size,:]
      yield batch_x, batch_y
  
  def net_train(self):
    '''
    网络训练
    '''
    epoches = 20
    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
    n_samples = mnist.train.num_examples
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    self.build_input()
    [o_out, loss, train_step] = self.build_net()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      for epoch in range(epoches):
        avg_cost = 0
        for bat, [batch_x, batch_y] in enumerate(self.obtain_mini_batch(n_samples, train_x, train_y)):
          _,cost = sess.run([train_step,loss], feed_dict={self.x:batch_x, self.y:batch_y, self.keep_prob:0.75})
          avg_cost += cost / n_samples * self.batch_size
        print('epoches: {:>3}/{} - train loss: {:>6.3f}'
        .format(epoch+1, epoches, avg_cost))
      correct_pred = tf.equal(tf.argmax(self.y,axis=1), tf.argmax(o_out,axis=1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      train_accuracy = sess.run(accuracy, feed_dict={self.x:train_x, self.y:train_y, self.keep_prob:1.0})
      print('train accuracy: {}'.format(train_accuracy))
      test_accuracy = sess.run(accuracy, feed_dict={self.x:test_x, self.y:test_y, self.keep_prob:1.0})
      print('test accuracy: {}'.format(test_accuracy))

MLP = MultiLayerPerceptron()
MLP.net_train()