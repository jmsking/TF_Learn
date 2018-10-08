#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import cifar10
import cifar10_input
import time

class ImprovedCNN():
	'''
	实现一个进阶的CNN
	识别CIFAR图像
	'''
	
	def __init__(self, img_h=24, img_w=24, channel=3, n_hidden1=384, n_hidden2=192, n_output=10, batch_size=128, epoches = 1000, err_threshold = 1e-2):
		'''
		CNN网络的初始化
		Args:
			img_h: 原始图片高度(像素)
			img_w: 原始图片宽度(像素)
			channel: 原始图片通道数(RGB)
			n_hidden1: 第一全连接网络中的隐藏层节点数
			n_hidden2: 第二全连接网络中的隐藏层节点数
			n_output: 全连接网络中的输出层节点数
			batch_size: mini_batch大小
			epoches: 迭代次数
			err_threshold: 训练误差阈值,控制训练深度
		'''
		self.img_h = img_h
		self.img_w = img_w
		self.channel = channel
		self.n_hidden1 = n_hidden1
		self.n_hidden2 = n_hidden2
		self.n_output = n_output
		self.batch_size = batch_size
		self.epoches = epoches
		self.err_threshold = err_threshold
		self.x = tf.placeholder(tf.float32, [batch_size,img_h,img_w,channel])
		self.y = tf.placeholder(tf.int32, [batch_size])
		#self.keep_prob = tf.placeholder(tf.float32)
		
	def obtain_input(self):
		'''
		获取CNN网络的输入样本
		Returns:
			train_x: 训练集
			train_y: 训练集标记
			test_x: 测试集
			test_y: 测试集标记
		'''
		data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
		cifar10.maybe_download_and_extract()
		train_x, train_y = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=self.batch_size)
		test_x, test_y = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=self.batch_size)
		return train_x, train_y, test_x, test_y
		
	
	def init_weight_with_loss(self, shape, stddev, w_loss):
		'''
		权重初始化(是否添加L2正则项)
		Args:
			shape: 待初始化的权重形状	例如-> [A, B, C, D],A*B:卷积核大小; C:图片颜色通道; D:卷积核数量
			stddev: 权重初始化的标准差
			w_loss: 添加的L2正则项
		Returns:
			初始化后的权重
		'''
		_w = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
		if w_loss is not None:
			weight_loss = tf.multiply(tf.nn.l2_loss(_w), w_loss, name='weight_loss')
			tf.add_to_collection('losses', weight_loss)
		return tf.Variable(_w)
	
	def init_bias(self, conv_core_num, bias_value):
		'''
		偏置初始化
		Args:
			conv_core_num: 卷积核数量
			bias_value: 偏置值
		Returns:
			初始化后的偏置
		'''
		_b = tf.constant(bias_value, shape=[conv_core_num,])
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
			logits: 网络输出(未经过Softmax处理)

		'''
		
		# 构建第一层卷积层及池化层
		w_conv1 = self.init_weight_with_loss(shape=[5,5,3,64],stddev=5e-2,w_loss=0.0)
		b_conv1 = self.init_bias(conv_core_num=64,bias_value=0.0)
		h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv_layer(self.x,w_conv1),b_conv1)) #卷积输出 
		h_pool1 = self.pool_layer(h_conv1, [1,3,3,1]) #池化
		norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
		
		#构建第二层卷积层及池化层
		w_conv2 = self.init_weight_with_loss(shape=[5,5,64,64], stddev=5e-2, w_loss=0.0)
		b_conv2 = self.init_bias(conv_core_num=64,bias_value=0.1)
		h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv_layer(h_pool1,w_conv2),b_conv2))
		norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
		h_pool2 = self.pool_layer(norm2, [1,3,3,1])
		
		# 经过了两次卷积层及池化层后,将得到的结果展平从而进入下面的全连接层
		h_pool2_flat = tf.reshape(h_pool2, [self.batch_size, -1])
		dim = h_pool2_flat.get_shape()[1].value # 节点数
		
		# 第二层的池化层结果作为之后的输入层,然后与隐藏层进行全连接
		w_fconn1 = self.init_weight_with_loss(shape=[dim, self.n_hidden1], stddev=0.04, w_loss=0.04)
		b_fconn1 = self.init_bias(conv_core_num=self.n_hidden1, bias_value=0.1)
		h_fconn1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fconn1) + b_fconn1) #隐藏层的输出
		
		# 将上一个全连层的结果再次经过全连接
		w_fconn2 = self.init_weight_with_loss(shape=[self.n_hidden1, self.n_hidden2], stddev=0.04, w_loss=0.04)
		b_fconn2 = self.init_bias(conv_core_num=self.n_hidden2, bias_value=0.1)
		h_fconn2 = tf.nn.relu(tf.matmul(h_fconn1,w_fconn2) + b_fconn2) #隐藏层的输出
		
		# 构建全连接中隐藏层与输出层之间的连接
		w_fconn3 = self.init_weight_with_loss(shape=[self.n_hidden2, self.n_output], stddev=1.0/192.0, w_loss=0.0)
		b_fconn3 = self.init_bias(conv_core_num=self.n_output, bias_value=0.1)
		logits = tf.add(tf.matmul(h_fconn2, w_fconn3),b_fconn3)
		
		return logits
		
	def obtain_loss(self, logits, labels):
		'''
		定义损失
		Args:
			logits: 网络输出
			labels: 真实标记
		Returns:
			网络总损失
		'''
		labels = tf.cast(labels, tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits, labels=labels, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		tf.add_to_collection('losses', cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'), name='total_loss')
		
	def obtain_test_accuracy(self, sess, top_k_op, test_x, test_y):
		'''
		获取测试准确率
		Args:
			sess: 当前计算图Session
			top_k_op: 准确率计算操作
			test_x: 测试集
			test_y: 测试集标记
		Returns:
			accuracy: 准确率
		'''
		n_samples = test_x.get_shape()[0].value
		max_iter = n_samples // self.batch_size
		total_samples = max_iter * self.batch_size
		true_count = 0
		iter = 0
		while iter < max_iter:
			batch_test_x, batch_test_y = sess.run([test_x, test_y])
			predicts = sess.run([top_k_op], feed_dict={self.x:batch_test_x, self.y:batch_test_y})
			true_count += np.sum(predicts)
			iter += 1
		accuracy = true_count / total_samples
		return accuracy
			
	
	def net_train(self):
		'''
		训练网络
		'''
		train_x, train_y, test_x, test_y = self.obtain_input()
		logits = self.build_net()
		loss = self.obtain_loss(logits, self.y)
		train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
		top_k_op = tf.nn.in_top_k(logits, self.y, 1)
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			# 获得协调对象
			coord = tf.train.Coordinator()
			# 启动队列
			tf.train.start_queue_runners(sess=sess, coord=coord)
			for epoch in range(self.epoches):
				start_time = time.time()
				batch_train_x, batch_train_y = sess.run([train_x, train_y])
				_, cost = sess.run([train_op, loss], feed_dict={self.x:batch_train_x, self.y:batch_train_y})
				duration = time.time() - start_time
				if epoch % 10 == 0:
					example_per_sec = self.batch_size / duration
					sec_per_batch = float(duration)
					format_str = 'epoch: {} - loss: {:>6.2f} - {:>6.1f}(example/sec) - {:>6.1f}(sec/batch)'
					print(format_str.format(epoch, cost, example_per_sec, sec_per_batch))
			coord.request_stop() # 请求线程结束
			coord.join() # 等待线程结束
			accuracy = self.obtain_test_accuracy(sess, top_k_op, test_x, test_y)
			print('Test accuracy: {:>6.2f}'.format(accuracy))
			
if __name__ == '__main__':
	cnn = ImprovedCNN()
	cnn.net_train()