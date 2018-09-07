#! /usr/bin/python3

import tensorflow as tf
import numpy as np

t_a = tf.Variable(np.arange(0,15).reshape(5,3))
t_b = tf.Variable(np.arange(0,6).reshape(2,3))
t_c = tf.Variable(np.arange(0,12).reshape(4,3))
# 单tensor
#embeddings = [t_a]
# 多tensor
embeddings = [t_a, t_b, t_c]
#input_ids = [[2,3],[1,1],[5,4]]
input_ids = [10]
embed_input = tf.nn.embedding_lookup(embeddings, input_ids, partition_strategy = 'div') # partition_strategy = 'mod'
embed_input_unique = tf.contrib.layers.embedding_lookup_unique(embeddings, input_ids)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("embeddings length: \n", len(sess.run(embeddings)))
	print("embeddings: \n", sess.run(embeddings))
	print("embed_input: \n", sess.run(embed_input))
	print("embed_input_unique: \n", sess.run(embed_input_unique))