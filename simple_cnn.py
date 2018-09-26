#! /usr/bin/python3

import tensorflow as tf
import input_data

class SimpleCNN():
	'''
	实现一个简单的CNN
	识别MNIST手写字体
	共两个卷积层,两个池化层及Softmax分类层
	'''
	
	def __init__(self, img_h=28, img_w=28, n_hidden=1024, n_output=10, batch_size=128, epoches = 2000):
		'''
		CNN网络的初始化
		Args:
			img_h: 原始图片高度(像素)
			img_w: 原始图片宽度(像素)
			n_hidden: 全连接网络中的隐藏层节点数
			n_output: 全连接网络中的输出层节点数
			batch_size: mini_batch大小
			epoches: 迭代次数
		'''
		self.img_h = img_h
		self.img_w = img_w
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.batch_size = batch_size
		self.epoches = epoches
		self.x = tf.placeholder(tf.float32, [None,img_h*img_w])
		self.y = tf.placeholder(tf.float32, [None,n_output])
		self.keep_prob = tf.placeholder(tf.float32)
		
	def obtain_input(self):
		'''
		获取CNN网络的输入样本
		Returns:
			train_x: 训练集
			train_y: 训练集标记
			test_x: 测试集
			test_y: 测试集标记
		'''
		mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
		train_x = mnist.train.images
		train_y = mnist.train.labels
		test_x = mnist.test.images
		test_y = mnist.test.labels
		return train_x, train_y, test_x, test_y
		
	
	def init_weight(self, shape):
		'''
		权重初始化
		Args:
			shape: 待初始化的权重形状	例如-> [A, B, C, D],A*B:卷积核大小; C:图片颜色通道; D:卷积核数量
		Returns:
			初始化后的权重
		'''
		_w = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(_w)
	
	def init_bias(self, conv_core_num):
		'''
		偏置初始化
		Args:
			conv_core_num: 卷积核数量
		Returns:
			初始化后的偏置
		'''
		_b = tf.constant(0.1, shape=[conv_core_num,])
		return tf.Variable(_b)
		
	def conv_layer(self, x, w):
		'''
		构建卷积层
		Args:
			x: 输入图片集,格式为: [N, H, W, C], N:图片数量; H:图片高度; W:图片宽度; C:图片颜色通道
			w: 卷积核权重,格式为: [A, B, C, D], A*B:卷积核大小; C:图片颜色通道; D:卷积核数量
		Returns:
			conv: 卷积层
		'''
		conv = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
		return conv
		
	def pool_layer(self, x, shape):
		'''
		构建池化层
		Args:
			x: 输入图片集,格式为: [N, H, W, C], N:图片数量; H:图片高度; W:图片宽度; C:图片颜色通道
			shape: 降采样维度,格式为[Dn, Dh, Dw, Dc], 分别表示对x的不同维度的Down-Sampling
		Returns:
			pool: 池化层
		'''
		pool = tf.nn.max_pool(x, ksize=shape, strides=[1,2,2,1], padding='SAME')
		return pool
		
	def build_net(self):
		'''
		构建网络结构
		Returns:
			y_: 网络输出
		'''
		# 将1D数据展开成2D数据
		x_img = tf.reshape(self.x, [-1, self.img_h, self.img_w, 1])
		
		# 构建第一层卷积层及池化层
		w_conv1 = self.init_weight([5,5,1,32])
		b_conv1 = self.init_bias(32)
		h_conv1 = tf.nn.relu(self.conv_layer(x_img,w_conv1) + b_conv1) #卷积输出 
		h_pool1 = self.pool_layer(h_conv1, [1,2,2,1]) #池化
		
		#构建第二层卷积层及池化层
		w_conv2 = self.init_weight([5,5,32,64])
		b_conv2 = self.init_bias(64)
		h_conv2 = tf.nn.relu(self.conv_layer(h_pool1,w_conv2) + b_conv2)
		h_pool2 = self.pool_layer(h_conv2, [1,2,2,1])
		
		# 经过了两次池化层,且每次的最大池化步长为2*2,最后图片大小为原来的1/4
		img_h = self.img_h // 4
		img_w = self.img_w // 4
		
		# 第二层的池化层结果作为之后的输入层,然后与隐藏层进行全连接
		w_fconn1 = self.init_weight([img_h*img_w*64, self.n_hidden])
		b_fconn1 = self.init_bias(self.n_hidden)
		h_pool2_flat = tf.reshape(h_pool2, [-1,img_h*img_w*64]) #将第二层池化层的结果展平
		h_fconn1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fconn1) + b_fconn1) #隐藏层的输出
		h_fconn1_drop = tf.nn.dropout(h_fconn1, self.keep_prob) # Dropout防止过拟合,同时降低计算复杂度
		
		# 构建全连接中隐藏层与输出层之间的连接
		w_fconn2 = self.init_weight([self.n_hidden, self.n_output])
		b_fconn2 = self.init_bias(self.n_output)
		y_ = tf.nn.softmax(tf.matmul(h_fconn1_drop, w_fconn2) + b_fconn2)
		
		return y_
		
	def obtain_mini_batch(self, train_x, train_y):
		'''
		获取Mini-batch
		Args:
			train_x: 训练集
			train_y: 训练集标记
		Returns:
			batch_train_x: mini-batch 训练集
			batch_train_y: mini-batch 训练集标记
		'''
		batches = len(train_x) // self.batch_size
		for i in range(batches):
			start = i*self.batch_size
			batch_train_x = train_x[start:start+self.batch_size]
			batch_train_y = train_y[start:start+self.batch_size]
			yield batch_train_x, batch_train_y
		
	def net_train(self):
		'''
		训练网络
		'''
		train_x, train_y, test_x, test_y = self.obtain_input()
		y_ = self.build_net()
		loss = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(y_), 1))
		train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
		predicts = tf.equal(tf.argmax(self.y), tf.argmax(y_, 1))
		accuracy = tf.cast(predicts, tf.float32)
		with tf.Session() as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			for epoches in range(200):
				for bat,(batch_train_x, batch_train_y) in enumerate(self.obtain_mini_batch(train_x, train_y)):
					_, cost = sess.run([train_op, loss], feed_dict={self.x:batch_train_x, self.y:batch_train_y, self.keep_prob:0.5})
					if bat % 100 == 0:
						train_loss = sess.run(loss, feed_dict={self.x:batch_train_x, self.y:batch_train_y, self.keep_prob:1})
						print('train loss {:>6.3f}'.format(train_loss))
			test_loss = sess.run(loss, feed_dict={self.x:test_x, self.y:test_y, self.keep_prob:1})
			print('test loss {:>6.3f}'.format(test_loss))
			
if __name__ == '__main__':
	cnn = SimpleCNN()
	cnn.net_train()