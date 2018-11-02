import numpy as np
import tensorflow as tf

num_epoch       = 10
num_samples     = 1000                              # 1000 for entire dataset
train_test_split      = 0.8
train_size    = int(num_samples * train_test_split)
test_size     = num_samples - train_size


if __name__ == '__main__':
    # Our net:
    labels = np.load("labels.npy")
    data = np.load("data.npy")#, dtype=np.ndarray)

    y_train = data[:train_size]
    x_train = labels[:train_size]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_train = x_train[0:train_size]
    x_test = x_train[train_size:]

    y_train = y_train[0:train_size]
    y_test = y_train[train_size:]


    model.fit(y_train, x_train, epochs=num_epoch)
    model.evaluate(y_test, y_test)