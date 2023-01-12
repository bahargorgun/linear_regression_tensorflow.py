import tensorflow as tf

# Define a simple linear regression model
x = tf.placeholder(tf.float32, shape=(None, 1))
y = tf.placeholder(tf.float32, shape=(None, 1))
w = tf.Variable(tf.random_normal([1, 1]))
b = tf.Variable(tf.random_normal([1]))
pred = tf.matmul(x, w) + b

# Define the loss function and optimizer
loss = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, current_loss = sess.run([train_op, loss], feed_dict={x: x_train, y: y_train})
        if i % 100 == 0:
            print("Step: {}, Loss: {}".format(i, current_loss))
    print("Final model parameters: w = {}, b = {}".format(sess.run(w), sess.run(b)))
