from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import datetime
def load_data():

    mnist = input_data.read_data_sets('./Data', one_hot=False)
    xTrain, labels = mnist.train.next_batch(50000)
    xTest, labels_test = mnist.train.next_batch(10000)
    data = xTrain * 255
    h = np.shape(xTrain)[0]
    w = np.shape(xTrain)[1]
    xTrain = np.reshape(xTrain, (h, w, 1))
    h = np.shape(xTest)[0]
    w = np.shape(xTest)[1]
    xTest = np.reshape(xTest, (h, w, 1))
    yTrain = np_utils.to_categorical(labels, 10)
    yTest = np_utils.to_categorical(labels_test, 10)
    return xTrain, yTrain, xTest, yTest, data, labels
def plot_sample_img(data, labels):
    pos = 1
    plt.figure(figsize=(4, 5))
    for i in range(10):
        # クラスiの画像のインデックスリストを取得
        targets = np.where(labels == i)[0]
        np.random.shuffle(targets)
        # 最初の10枚の画像を描画
        for idx in targets[:8]:
            plt.subplot(10, 8, pos)
            img = data[idx]
            # (channel, row, column) => (row, column, channel)
            plt.imshow(img.reshape(28, 28))
            plt.axis('off')
            pos += 1
    plt.show()
def model(param):

    #  モデル定義
    model = Sequential()
    model.add(Conv1D(param['filter'][0], param['kanel_conv'][0], padding='same',
                     input_shape=(param['width'], 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(param['kanel_pool'][0], padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(param['drop'][0]))

    model.add(Conv1D(param['filter'][1], param['kanel_conv'][1], padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(param['kanel_pool'][1], padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(param['drop'][1]))

    model.add(Flatten())
    model.add(Dense(param['dense'][0]))
    model.add(BatchNormalization())
    model.add(Dropout(param['drop'][2]))
    model.add(Dense(param['dense'][1]))
    model.add(BatchNormalization())
    model.add(Dropout(param['drop'][3]))
    model.add(Dense(param['dense'][2]))
    model.add(BatchNormalization())
    model.add(Dropout(param['drop'][4]))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()

    return model
def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(5, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlim([-0.5, 9.5])
    plt.ylim([-0.5, 9.5])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def output_result(xTest, yTest_1, model, history, label_names, showfig=True):
    # accuracy
    yPredict = model.predict(xTest)
    yPred = [yPredict[i, :].argmax() for i in range(0, yPredict.shape[0])]
    yTest = [yTest_1[i, :].argmax() for i in range(0, yPredict.shape[0])]
    acc = accuracy_score(yTest, yPred)
    date = datetime.datetime.now()
    date = date.strftime('%Y%m%d-%H-%M')

    if showfig == True:
        print(acc)
        cnf_matrix = confusion_matrix(yTest, yPred)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=label_names, title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=label_names, normalize=True, title='Normalized confusion matrix')
        plt.savefig('keras/result/{}-acc.png'.format(date))

        plt.figure()
        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='loss', color='blue')
        plt.plot(range(len(history.history['acc'])), history.history['acc'], label='acc', color='yellow')
        plt.legend()
        plt.savefig('keras/result/{}-loss.png'.format(date))
xTrain, yTrain, xTest, yTest, data, labels = load_data()
plot_sample_img(data, labels)
param = {'epochs': 100,
         'batch': 512,
         'width': 28*28,
         'filter': [64, 32],
         'kanel_conv': [4, 4],
         'kanel_pool': [3, 3],
         'drop': [0.1, 0.1, 0.1, 0.1, 0.1],
         'dense': [11*2**8, 2**8, 10]}
model = model(param)
history = model.fit(xTrain, yTrain, epochs=param['epochs'], batch_size=param['batch'], )
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
output_result(xTest, yTest, model, history, label_names)


