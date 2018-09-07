#! /user/bin/python3

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

def embedding_lookup_unique(params, ids, sess, name=None):
	ids = ops.convert_to_tensor(ids)
	print(sess.run(ids))
	shape = array_ops.shape(ids)
	print(sess.run(shape))
	ids_flat = array_ops.reshape(
		ids, math_ops.reduce_prod(shape, keepdims=True))
	print("ids_flat: \n", sess.run(ids_flat))
	unique_ids, idx = array_ops.unique(ids_flat)
	print("unique_ids: \n", sess.run(unique_ids))
	print("idx: \n", sess.run(idx))
	unique_embeddings = embedding_ops.embedding_lookup(params, unique_ids)
	print("params: \n", sess.run(params))
	print("unique_embeddings: \n", sess.run(unique_embeddings))
	embeds_flat = array_ops.gather(unique_embeddings, idx)
	print("embeds_flat: \n", sess.run(embeds_flat))
	embed_shape = array_ops.concat(
		[shape, array_ops.shape(unique_embeddings)[1:]], 0)
	embeds = array_ops.reshape(embeds_flat, embed_shape)
	embeds.set_shape(ids.get_shape().concatenate(
		unique_embeddings.get_shape()[1:]))
	return embeds
	
t_a = tf.Variable(np.arange(0,15).reshape(5,3))
t_b = tf.Variable(np.arange(0,6).reshape(2,3))
t_c = tf.Variable(np.arange(0,12).reshape(4,3))
embeddings = [t_a, t_b, t_c]
input_ids = [[2,3],[1,1],[3,4]]
#embed_input_unique = tf.contrib.layers.embedding_lookup_unique(embeddings, input_ids)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("result: \n",sess.run(embedding_lookup_unique(embeddings, input_ids, sess)))