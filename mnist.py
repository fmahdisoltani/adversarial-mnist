"""
A softmax classifier for MNIST.
Based on TensorFlow tutorial on Deep MNIST.
"""

import tensorflow as tf


def main():
    sess = tf.InteractiveSession()

    #Load MNIST Data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    #Variables
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    y = tf.matmul(x,W) + b

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #Train the model
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for _ in range(1000):
      batch = mnist.train.next_batch(100)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    #Evaluate the model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main()