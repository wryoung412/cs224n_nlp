import numpy as np
import tensorflow as tf

# with tf.Session() as sess:
#     with tf.variable_scope("rnn"):
#         W_x = tf.get_variable('W_x', shape=(3, 2), dtype=tf.float32,
#                               initializer=tf.contrib.layers.xavier_initializer())
#         tf.get_variable_scope().reuse_variables()
#         W_x = tf.get_variable("W_x", initializer=np.array(np.eye(3,2), dtype=np.float32))
#         W_x = tf.get_variable("W_x", initializer=np.array(np.zeros([3,2]), dtype=np.float32))
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(W_x))

x = [1, 2]
x.extend([3, 4])
print([[0] * 5] * 3)
