from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


R = 1.3
dim = 500
learning_rate = 0.0001
num_steps = 1000000
batch_size = 50

# generate the datasets
dataset_x = []
dataset_y = []
for _ in range(50000000):
    x = np.random.normal(size=dim)
    x = x / np.linalg.norm(x, ord=2)
    y = np.random.choice([1, R])
    dataset_x.append(x * y)
    dataset_y.append([0, 1] if y == R else [1, 0])


def sample_data(size):
    for j in range(0, len(dataset_x), size):
        yield np.array(dataset_x[j: j+size), np.array(dataset_y[j: j+size])


generator = sample_data(batch_size)

X = tf.placeholder("float", [None, dim])
Y = tf.placeholder("float", [None, 2])

weights = {
    'h1': tf.Variable(tf.random_normal([500, 1000])),
    'h2': tf.Variable(tf.random_normal([1000, 1000])),
    'out': tf.Variable(tf.random_normal([1000, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([1000])),
    'b2': tf.Variable(tf.random_normal([1000])),
    'out': tf.Variable(tf.random_normal([2]))
}


def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.contrib.layers.batch_norm(layer_1)
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.contrib.layers.batch_norm(layer_2)
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return tf.nn.relu(out_layer)


logits = neural_net(X)

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next(generator)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % 1000 == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("[INFO] Step {0:d}, Minibatch Loss = {1:.4f}, Training Accuracy = {2:.3f}".format(
                step, loss, acc))

    print("Training Finished!")
