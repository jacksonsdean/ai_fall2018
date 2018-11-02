import numpy as np
import tensorflow as tf

N_EPOCH = 10
N_SAMPLES = 1000
TRAIN_SPLIT = 0.8

train_size = int(N_SAMPLES * TRAIN_SPLIT)
test_size = N_SAMPLES - train_size


if __name__ == '__main__':
    # Our net:
    labels = np.load("labels.npy")
    data = np.load("data.npy")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_train = data[0:train_size]
    x_test = data[train_size:]

    y_train = labels[0:train_size]
    y_test = labels[train_size:]

    model.fit(x_train, y_train, epochs=N_EPOCH)
    model.evaluate(x_test, y_test)