from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Users/QYGong/Downloads/mnist", one_hot=True)

# print(mnist.train.images.shape, mnist.train.labels.shape)
# print(mnist.test.images.shape, mnist.test.labels.shape)
# print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuary.eval({x: mnist.test.images, y_: mnist.test.labels}))

# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])


# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))

# sess.run(tf.global_variables_initializer())
# y = tf.matmul(x,W) + b
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# for _ in range(1000):
#   batch = mnist.train.next_batch(100)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1]})
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))