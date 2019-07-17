from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data

n_sample_train = 5000
n_sample_test = 1000

def get_MNIST_data():

    mnist = input_data.read_data_sets('./Data', one_hot=True)
    train_x, one_hots_train = mnist.train.next_batch(n_sample_train)
    test_x, one_hots_test = mnist.train.next_batch(n_sample_test)

    return train_x, one_hots_train, test_x, one_hots_test
def plot_MNIST(x, one_hot):

    row = 4
    column = 4
    p = [0, 12, 54, 45, 17, 12, 45, 89, 154, 111, 156, 8, 13, 14, 15, 16]

    plt.figure()

    for i in range(row * column):

        image = x[p[i]].reshape(28, 28)
        plt.subplot(row, column, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title('label = {}'.format(np.argmax(one_hot[p[i]]).astype(int)))
        plt.axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                    wspace=0.05, hspace=0.3)
    plt.show()
def confusion_matrix(cm, accuracy):

    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)
def dense(input, name, in_size, out_size, activation="relu"):

    with tf.variable_scope(name, reuse=False):
        w = tf.get_variable("w", shape=[in_size, out_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(input, w), b)

        if activation == "relu":
            l = tf.nn.relu(l)
        elif activation == "sigmoid":
            l = tf.nn.sigmoid(l)
        elif activation == "tanh":
            l = tf.nn.tanh(l)
        else:
            l = l
        print(l)
    return l
def scope(y, y_, sess, learning_rate=0.1):

    #Learning rate
    learning_rate = tf.Variable(learning_rate,  trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_), name="loss")

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       name="optimizer").minimize(loss)

    # Evaluate the model
    correct = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int32),
                       tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    #  Tensorboard
    writer = tf.summary.FileWriter('./Tensorboard/')
    # run this command in the terminal to launch tensorboard:
    # tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return loss, accuracy, optimizer, writer
train_x, one_hots_train, test_x, one_hots_test = get_MNIST_data()
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class
height = train_x.shape[1]               # All the pixels are represented as a vector (dim: 784)

# Session and context manager
tf.reset_default_graph() # graph reset(grow of data)
sess = tf.Session() # quate a session
with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders(data type definition)
    x = tf.placeholder(tf.float32, [None, height, 1], name='X')
    y = tf.placeholder(tf.float32, [None, n_label], name='Y')
    # Neural network
    # Y = wX + b nonactivation
    # Y = f(wX + b) activation : sigmoid, tanh, reLU
    # sigmoid : f = 1/(1 + exp(-x))
    # tanh    : f = tanh(x)
    # reLU    : f = max(x, 0)
    size = x.get_shape().as_list()
    l0 = tf.reshape(x, [-1, size[1] * size[2]], name='reshape_to_fully')
    l1 = dense(l0, "layer1", height, 128)
    l2 = dense(l1, "layer2", 128, 128)
    l3 = dense(l2, "output_layer", 128, n_label, activation=None)
    # Softmax layer
    # f = exp(xi)/sigma(exp(xi))
    y_ = tf.nn.softmax(l3, name="softmax")
    # Scope
    loss, accuracy, optimizer, writer = scope(y, y_, sess, learning_rate=0.01)
    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())
    loss_history = []
    acc_history = []
    epoch = 200

    train_x = np.reshape(train_x, [-1, height, 1])
    train_data = {x: train_x, y: one_hots_train}
    for e in range(epoch):
        _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict=train_data)
        loss_history.append(l)
        acc_history.append(acc)
        print("Epoch" + str(e) + "_Loss" + str(l) + "_" + str(acc))
    # Train the Neural Network


# Test the trained Neural Network
test_x = np.reshape(test_x, [-1, height, 1])
test_data = {x: test_x, y: one_hots_test}
l, acc = sess.run([loss, accuracy], feed_dict=test_data)
print("Test - Loss: " + str(l) + " - " + str(acc))
predictions = y_.eval(feed_dict=test_data, session=sess)
predictions_int = (predictions == predictions.max(axis=1, keepdims=True)).astype(int)
predictions_numbers = [predictions_int[i, :].argmax() for i in range(0, predictions_int.shape[0])]
# Confusion matrix
cm = metrics.confusion_matrix(number_test, predictions_numbers) / 30
print(cm)
confusion_matrix(cm=cm, accuracy=acc)
cmN = cm / cm.sum(axis=0)
confusion_matrix(cm=cmN, accuracy=acc)