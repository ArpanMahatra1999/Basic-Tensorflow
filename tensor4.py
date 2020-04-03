import tensorflow as tf

with tf.compat.v1.Session() as sess:
    a = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    x = tf.compat.v1.placeholder(tf.float32)

    linear_model = a * x + b

    init = tf.compat.v1.global_variables_initializer()

    # sess.run(init)

    y = tf.compat.v1.placeholder(tf.float32)

    squared_deltas = tf.square(linear_model - y)

    loss = tf.reduce_sum(squared_deltas)

    # print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)

    train = optimizer.minimize(loss)

    sess.run(init)

    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([a, b]))
