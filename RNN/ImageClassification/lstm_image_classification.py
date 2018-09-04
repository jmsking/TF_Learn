#! /usr/bin/python3


import tensorflow as tf
import input_data
from tensorflow.contrib import rnn

class LSTMImageClassificatin():
	"""
	使用LSTM进行MNIST数字图像进行分类
	@author: MaySnow
	"""
		
	def __init__(self, n_hiddens, batch_size, time_steps, n_features, 
				n_classes, learn_rate, epoches, n_displays):
		"""
		类初始化方法.
		Args:
			n_hiddens: 每层LSTM层中的单元数, 格式: [n1, n2, ...]。
			n1,n2分别表示第一层、第二层中的LSTM单元数
			batch_size: 批次大小
			time_steps: 按时间展开的次数
			n_features: 样本属性数
			n_classes: 分类数
			learn_rate: 学习速率
			epoches: 迭代次数
		"""
		self._n_hiddens = n_hiddens
		self._batch_size = batch_size
		self._time_steps = time_steps
		self._n_features = n_features
		self._n_classes = n_classes
		self._learn_rate = learn_rate
		self._epoches = epoches
		self._n_displays = n_displays
		
		self.X = tf.placeholder(tf.float32, [None,time_steps,n_features])
		self.y = tf.placeholder(tf.float32, [None,n_classes])
		self.W = tf.Variable(tf.truncated_normal(shape=[n_hiddens[-1], n_classes]), dtype=tf.float32)
		self.b = tf.Variable(tf.constant(0.0, shape=[n_classes]), dtype=tf.float32)
		
	def _build_model(self):
		"""
		构建多层LSTM模型.
		Returns:
			LSTM模型
		"""
		
		def single_layer(num_units):
			#构建单层LSTM
			single_lstm_layer = rnn.BasicLSTMCell(num_units)
			return single_lstm_layer
		
		lstm_layers = [single_layer(n) for n in self._n_hiddens]
		lstm_model = rnn.MultiRNNCell(lstm_layers)
		return lstm_model
	
	def _obtain_lstm_output(self, lstm_model):
		'''
		获取LSTM模型的输出
		'''
		outputs, state = tf.nn.dynamic_rnn(lstm_model, self.X, dtype=tf.float32)
		h_state = state[-1][1]
		return h_state
		
	def _build_classification(self, inputs, n_inputs):
		"""
		构建分类层
		""" 
		_y = tf.nn.softmax(tf.add(tf.matmul(inputs, self.W), self.b))
		return _y
	
	def _obtain_mini_batch(self, train_x, train_y):
		"""
		获取Mini-batch训练样本
		"""
		_len, _, _ = train_x.shape
		batches = _len // self._batch_size
		for i in range(batches):
			start = i*self._batch_size
			end = start + self._batch_size
			batch_train_x, batch_train_y = train_x[start:end], train_y[start:end]
			yield batch_train_x, batch_train_y
		
		
	def start_train(self):
		"""
		开始训练
		"""
		mnist = input_data.read_data_sets('MNIST', one_hot = True)
		train_x = mnist.train.images
		train_y = mnist.train.labels
		test_x = mnist.test.images
		test_y = mnist.test.labels
		train_x = train_x.reshape(-1,self._time_steps,self._n_features)
		test_x = test_x.reshape(-1,self._time_steps,self._n_features)
		print(train_x.shape, '\t', train_y.shape)
		lstm_model = self._build_model()
		h_state = self._obtain_lstm_output(lstm_model)
		_y = self._build_classification(h_state, self._n_hiddens[-1])
		cross_entropy = -tf.reduce_sum(self.y*tf.log(_y))
		train_op = tf.train.AdamOptimizer(self._learn_rate).minimize(cross_entropy)
		predicts = tf.equal(tf.argmax(_y,1),tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(predicts, 'float'))
		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)
			for epoch in range(self._epoches):
				cost_list = []
				for _, [batch_train_x, batch_train_y] in enumerate(self._obtain_mini_batch(train_x, train_y)):
					_, cost = sess.run([train_op, cross_entropy],feed_dict={self.X:batch_train_x,
							self.y:batch_train_y})
					cost_list.append(cost / self._batch_size)
				avg_cost = sum(cost_list)/len(cost_list)
				if epoch % self._n_displays == 0:
					print('Epoch: {:>6d}/{:>6d} - cost: {:>6.2f}'.format(epoch,self._epoches, avg_cost))
			print(sess.run(accuracy, feed_dict={self.X:train_x, self.y:train_y}))
			print(sess.run(accuracy, feed_dict={self.X:test_x, self.y:test_y}))
				
print('------start------')
n_hiddens, batch_size = [80, 50], 128
time_steps, n_features, n_classes = 1, 28*28, 10
learn_rate, epoches, n_displays = 0.005, 10, 2
lstm = LSTMImageClassificatin(n_hiddens, batch_size, time_steps, n_features, 
		n_classes, learn_rate, epoches, n_displays)
lstm.start_train()