from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import os
from sklearn.neighbors import KNeighborsClassifier
def hello():
    print('Hello git-hub !')
    print('Hello git-hub !')


hello()

# the number of sumple(train and test)
n_sample_train = 50000
n_sample_test = 10000


def Latest_space_plot(encoded_train, one_hots_train, one_hots_test):
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1)

    cmap = cm.get_cmap('jet', one_hots_train.shape[1])
    colors = []
    label_train = np.argmax(one_hots_train, axis=1)
    label_test = np.argmax(one_hots_test, axis=1)
    for i in range(cmap.N):
        colors.append(cmap(i))
    for i in range(one_hots_train.shape[1]):
        ind = np.where(label_train == i)
        ax.scatter(encoded_train[ind, 0], encoded_train[ind, 1],
                   color=colors[i], s=5, label=str(i) + '_train')
    for i in range(one_hots_test.shape[1]):
        ind = np.where(label_test == i)
        ax.scatter(encoded_test[ind, 0], encoded_test[ind, 1],
                   color=colors[i], s=100, label=str(i) + '_test')
        ax.legend()

def get_MNIST_data():


    mnist = input_data.read_data_sets('./Data', one_hot=True)
    train_x, one_hots_train = mnist.train.next_batch(n_sample_train)
    test_x, one_hots_test = mnist.train.next_batch(n_sample_test)

    return train_x, one_hots_train, test_x, one_hots_test

def plot_MNIST(x, one_hot):

    row = 4
    column = 4
    p = random.sample(range(1, 100), row * column)

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

def dense(inputs, in_size, out_size, activation='sigmoid', name='layer'):

    with tf.variable_scope(name, reuse=False):

        w = tf.get_variable("w", shape=[in_size, out_size], initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(inputs, w), b)

        if activation == 'relu':
            l = tf.nn.relu(l)
        elif activation == 'sigmoid':
            l = tf.nn.sigmoid(l)
        elif activation == 'tanh':
            l = tf.nn.tanh(l)
        elif activation == 'leaky_relu':
            l = tf.nn.leaky_relu(l)
        else:
            l = l


    return l

def confusion_matrix(cm, accuracy):

    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)

train_x, one_hots_train, test_x, one_hots_test = get_MNIST_data()
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class(extract original number list and check the length)
height = train_x.shape[1]               # All the pixels are represented as a vector (dim: 784)

z_dimension = 2 # Latent space dimension

# Hyperparameters
hyperparameters_encode = {'en_size': [height, 196, 49, 7, z_dimension],
                         'en_activation': ['relu', 'relu', 'relu', 'linear'],
                         'names': ['en_layer_1', 'en_layer_2', 'en_layer_3', 'latent_scope']}
hyperparameters_decode = {'de_size': [z_dimension, 7, 49, 196, height],
                         'de_activation': ['relu', 'relu', 'relu', 'linear'],
                         'names': ['de_layer_1', 'de_layer_2', 'de_layer_3', 'de_layer_out']}
hyperparameters_scope = {'learning_rate': 0.01, 'maxEpoch': 200, 'batch_size': 200}

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, height], name='X')

    # Encoder
    print('')
    print("ENCODER")
    print(x)
    l1 = dense(x, in_size=hyperparameters_encode['en_size'][0],
               out_size=hyperparameters_encode['en_size'][1],
               activation=hyperparameters_encode['en_activation'][0],
               name=hyperparameters_encode['names'][0])
    print(l1)
    l2 = dense(l1, in_size=hyperparameters_encode['en_size'][1],
               out_size=hyperparameters_encode['en_size'][2],
               activation=hyperparameters_encode['en_activation'][1],
               name=hyperparameters_encode['names'][1])
    print(l2)
    l3 = dense(l2, in_size=hyperparameters_encode['en_size'][2],
              out_size=hyperparameters_encode['en_size'][3],
              activation=hyperparameters_encode['en_activation'][2],
              name=hyperparameters_encode['names'][2])
    print(l3)
    z = dense(l3, in_size=hyperparameters_encode['en_size'][3],
              out_size=hyperparameters_encode['en_size'][4],
              activation=hyperparameters_encode['en_activation'][3],
              name=hyperparameters_encode['names'][3])
    print(z)

    # Decoder
    print('')
    print("DECODER")
    r1 = dense(z, in_size=hyperparameters_decode['de_size'][0],
               out_size=hyperparameters_decode['de_size'][1],
               activation=hyperparameters_decode['de_activation'][0],
               name=hyperparameters_decode['names'][0])
    print(r1)
    r2 = dense(r1, in_size=hyperparameters_decode['de_size'][1],
               out_size=hyperparameters_decode['de_size'][2],
               activation=hyperparameters_decode['de_activation'][1],
               name=hyperparameters_decode['names'][1])
    print(r2)
    r3 = dense(r2, in_size=hyperparameters_decode['de_size'][2],
               out_size=hyperparameters_decode['de_size'][3],
               activation=hyperparameters_decode['de_activation'][2],
               name=hyperparameters_decode['names'][2])
    print(r3)
    x_hut = dense(r3, in_size=hyperparameters_decode['de_size'][3],
                  out_size=hyperparameters_decode['de_size'][4],
                  activation=hyperparameters_decode['de_activation'][3],
                  name=hyperparameters_decode['names'][3])
    print(x_hut)
    # Scope
    learning_rate = tf.Variable(hyperparameters_scope['learning_rate'], trainable=False)

    # Loss
    loss = tf.reduce_mean(tf.square(x-x_hut))

    # Train the Neural Network
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)

    # Tensorboard
    writer = tf.summary.FileWriter('.\Tensorboard')
    writer.add_graph(graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history = []
    for epoch in range(hyperparameters_scope['maxEpoch']):
        i = 0
        loss_batch = []
        while i < n_sample_train:

            start = i
            end = i + hyperparameters_scope['batch_size']

            train_data = {x: train_x[start: end]}
            _, l = sess.run([optimizer, loss], feed_dict=train_data)

            loss_batch.append(l)
            i = i + hyperparameters_scope['batch_size']
        epoch_loss = np.mean(loss_batch)
        loss_history.append(epoch_loss)
        print('Epoch', epoch, '/', hyperparameters_scope['maxEpoch'], ' : loss', epoch_loss)

    # Encode the training data
    train_data = {x: train_x}
    encoded_train = sess.run(z, feed_dict=train_data)
    number_train = [one_hots_train[i, :].argmax() for i in range(0, one_hots_train.shape[0])]
    test_data = {x: test_x}
    encoded_test = sess.run(z, feed_dict=test_data)
    number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]
    # plt.figure()
    # plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=number_train)
    # plt.figure()
    # plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=number_test)
    Latest_space_plot(encoded_train, one_hots_train, one_hots_test)
    # Reconstruct the data at the output of the decoder
    reconstructed = sess.run(x_hut, feed_dict=train_data)

    nn = 10
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(encoded_train, np.argmax(one_hots_train, axis=1))
    Label_test_predicited = neigh.predict(encoded_test)
    Label_test_true = np.argmax(one_hots_test, axis=1)

    score = 0
    for i in range(one_hots_test.shape[0]):
        if Label_test_predicited[i] == Label_test_true[i]:
            score += 1
    accuracy =score/np.round(one_hots_test.shape[0], 4)
    print(accuracy)
# Plot the latent space

# Plot reconstruction

# PCA
