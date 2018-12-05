import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

n_epoch = 1
N_SAMPLES = 1000
TRAIN_SPLIT = 0.8

N_CATEGORIES = 10

batch_size = 10
train_size = int(N_SAMPLES * TRAIN_SPLIT)
test_size = N_SAMPLES - train_size


class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        try:
            batch_size = int(sys.argv[1])
            n_epoch = int(sys.argv[2])
        except ValueError:
            print("Please use ints for batch size and number of epochs")
            exit()

    elif len(sys.argv) == 2:
        print("Please provide both batch size and number of epochs")
        exit()
    elif len(sys.argv) > 3:
        print("Please only provide batch size and number of epochs")
        exit()



    # Our net:
    labels = np.load("labels.npy")
    data = np.load("data.npy")
    data = np.ndarray.astype(data, "float32")
    labels = np.ndarray.astype(labels, "int")

    input_shape = (96, 1366,1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(4, 4),
            strides=(1, 1),  # X and Y to move the window by
            activation=tf.nn.relu,
            input_shape=input_shape),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(N_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # WHY DID WE DO THIS??
                  metrics=['accuracy'])


    x_train = data[0:train_size]
    x_test = data[train_size:]

    x_train = x_train.reshape([-1, 96, 1366, 1])
    x_test = x_test.reshape([-1, 96, 1366, 1])


    y_train = labels[0:train_size]
    y_test = labels[train_size:]


    print("x:", x_train.shape)
    print("y:", len(y_train))

    # model.fit(x_train, y_train, epochs=N_EPOCH)#, steps_per_epoch=800)
    history = AccuracyHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=n_epoch,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])

    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(range(1, 11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


