"""
A Multilayer Convolutional Network for MNIST.
Based on TensorFlow tutorial on Deep MNIST.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def run_conv2d(x_image, keep_prob):
    """
    Args:
        x_image: shape [-1, 28, 28, 1]
        keep_prob:  controls the dropout rate
    Returns: y_conv the output of the conv net

    """

    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv


def train_and_eval(session, mnist):
    # Train and and Evaluate the Model

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = session.run(
                accuracy,
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


    print("test accuracy %g" % session.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    saver = tf.train.Saver()
    save_path = saver.save(session, "./model.ckpt")
    print("Model saved in file: %s" % save_path)


def extract_images_of(mnist, digit):
  """
  Extract all the images of digit from the input data mnist
  Args:
      mnist: MNIST data
      digit: indicates which digit we are interested in to extract,
             digit should be a number in [0,9]

  Returns: the images of 'digit' from the input data mnist
  """

  result = []
  for i in range (mnist.train.num_examples - 1):
      if mnist.train.labels[i, digit] == 1:
          result.append(mnist.train.images[i:i + 1,:])

  return result

def one_hot(digit):
    """
    Return the one hot encoding for digit
    Args:
        digit: a number in the range [0,9]

    Returns: one hot encoding of digit

    """
    result = np.zeros(10)
    result[digit] = 1
    return result

def make_adversarial_images(session, mnist, source_digit, dest_digit):
    """
    Create adversarial images from source_digit which are classified as dest_digit.
    Args:
        session: which session of tf should the function use
        mnist: MNIST data

    Returns: None
    """
    images_of_source = extract_images_of(mnist, source_digit)
    correctly_classified = 0
    fig = plt.figure(figsize=(4, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])

    for i in range(mnist.train.num_examples):
        print i
        [correct_prediction_res] = session.run([correct_prediction],
        feed_dict={x: images_of_source[i], y_:np.array(one_hot(source_digit)).reshape(1,10), keep_prob: 1})

        #pick an image of source which is correctly classified by the model
        if correct_prediction_res == 1:
            correctly_classified += 1
            tweak = True
            _ = session.run([initialization],feed_dict={x: images_of_source[i]})

            while tweak:
                [pred, _] = session.run(
                [correct_prediction_b, train_step_b],
                feed_dict={ y_: np.array(one_hot(dest_digit)).reshape(1, 10), keep_prob: 1 })
                tweak = not pred[0]


            [adversary_image] = session.run([adversary_xvar])

            ax = fig.add_subplot(10, 3, 3 * correctly_classified - 2)
            plt.imshow(images_of_source[i].reshape(28, 28), cmap='gray')
            ax.set_yticklabels([])
            ax.set_xticklabels([])



            ax = fig.add_subplot(10, 3, 3 * correctly_classified - 1 )
            plt.imshow(images_of_source[i].reshape(28, 28) - adversary_image.reshape(28, 28), cmap='gray')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            ax = fig.add_subplot(10, 3, 3 * correctly_classified )
            plt.imshow(adversary_image.reshape(28,28), cmap='gray')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            if correctly_classified == 10:
                break


    # plt.show()

    plt.savefig('adversarial_images')
    plt.show()

def build_graph(x, y_, keep_prob, vars_to_optimize = None):
    """

    Args:
        x: input data
        y_: input labels
        keep_prob: the parameter that controls the drop_out rate
        vars_to_optimize: the variables that are tuned during optimization

    Returns:

    """
    y_conv = run_conv2d(x, keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=vars_to_optimize)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step, correct_prediction, accuracy


#Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
adversary_xvar = tf.Variable(tf.zeros([1, 28, 28, 1]), dtype=tf.float32)

initialization = tf.assign(adversary_xvar, tf.reshape(x, [1, 28, 28, 1]))
[train_step, correct_prediction, accuracy] = build_graph(x_image, y_, keep_prob)
[train_step_b, correct_prediction_b, _] = build_graph(adversary_xvar, y_, keep_prob, [adversary_xvar])


def main():
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    #Load MNIST Data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    train_and_eval(sess, mnist)
    make_adversarial_images(sess, mnist, 2, 6)


if __name__ == '__main__':
    main()