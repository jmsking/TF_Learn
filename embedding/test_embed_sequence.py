#! /usr/bin/python3

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import embedding_ops as contrib_embedding_ops
from tensorflow.python.ops import embedding_ops

def model_variable(name, sess, shape=None, dtype=dtypes.float32, initializer=None,
						regularizer=None, trainable=True, collections=None,
						caching_device=None, device=None, partitioner=None,
						custom_getter=None, use_resource=None):

	collections = list(collections or [])
	print(sess.run(collections))
	collections += [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
	print(sess.run(collections))
	var = variable(name, shape=shape, dtype=dtype,
						initializer=initializer, regularizer=regularizer,
						trainable=trainable, collections=collections,
						caching_device=caching_device, device=device,
						partitioner=partitioner, custom_getter=custom_getter,
						use_resource=use_resource)
	return var

	
def embed_sequence(ids,
					sess,	#添加
					vocab_size=None,
					embed_dim=None,
					unique=False,
					initializer=None,
					regularizer=None,
					trainable=True,
					scope=None,
					reuse=None):

	if not (reuse or (vocab_size and embed_dim)):
		raise ValueError('Must specify vocab size and embedding dimension when not '
							'reusing. Got vocab_size=%s and embed_dim=%s' % (
							vocab_size, embed_dim))
	with variable_scope.variable_scope(
			scope, 'EmbedSequence', [ids], reuse=reuse):
		shape = [vocab_size, embed_dim]
		if reuse and vocab_size is None or embed_dim is None:
			shape = None
		embeddings = model_variable(
				'embeddings', 
				sess,	#添加
				shape=shape,
				initializer=initializer, regularizer=regularizer,
				trainable=trainable)
		print(sess.run(embeddings))
		if unique:
			return contrib_embedding_ops.embedding_lookup_unique(embeddings, ids)
		return embedding_ops.embedding_lookup(embeddings, ids)	

	
inputs = tf.constant([5, 1, 3, 2, 1])
vocab_size = 10
embed_dim = 3
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	input_embeddings = embed_sequence(inputs, sess, vocab_size, embed_dim)
	print(sess.run(input_embeddings))