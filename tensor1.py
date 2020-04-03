import tensorflow as tf

with tf.compat.v1.Session() as ses:
# Build a graph.
    node1 = tf.constant(5.0, tf.float32)
    node2 = tf.constant(6.0)
    c = node1 * node2

# Evaluate the tensor `c`.
    print(ses.run(c))
