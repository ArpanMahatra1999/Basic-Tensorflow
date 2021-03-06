import tensorflow as tf

with tf.compat.v1.Session() as sess:
    a = tf.compat.v1.placeholder(tf.float32)
    b = tf.compat.v1.placeholder(tf.float32)

    adder_node = a + b

    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
