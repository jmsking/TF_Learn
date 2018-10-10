#! /usr/bin/python3

import math
import numpy as np
import tensorflow as tf
"""
logit 形如 ln(p/(1-p))
"""

labels = [[0.2,0.3,0.5],[0.1,0.6,0.3]]
logits = [[4,1,-2],[0.1,1,3]]
#labels = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]))
#logits = tf.convert_to_tensor(np.array([[0.5, 1.5, 0.1],[2.2, 1.3, 1.7]]))
logits_scaled = tf.nn.softmax(logits)
result = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
x = labels*tf.log(logits_scaled)
cross_entroy = -tf.reduce_sum(x, 1)
#print(1*math.log(0.61939586, math.e))
with tf.Session() as sess:    
	print(sess.run(logits_scaled))    
	print(sess.run(result))
	print(sess.run(x))
	print(sess.run(cross_entroy))
	
