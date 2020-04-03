import tensorflow as tf

with tf.compat.v1.Session() as sess:
    a = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    x = tf.compat.v1.placeholder(tf.float32)

    linear_model = a * x + b

    init = tf.compat.v1.global_variables_initializer()

    sess.run(init)

    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
