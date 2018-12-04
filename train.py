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
    data = np.ndarray.astype(data, "float32")
    labels = np.ndarray.astype(labels, "float32")

    shaped_data = np.array(data)
    shaped_labels = labels

    # for d in data:
    #     d1 = d.reshape(96,1366)
    #     shaped_data.append(d1)

    shaped_data = np.array(shaped_data)

    # shaped_data = tf.contrib.layers.batch_norm(shaped_data)
    # shaped_labels = tf.contrib.layers.batch_norm(labels)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(96, 1366)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # WHY DID WE DO THIS??
                  metrics=['accuracy'])



    shaped_data = shaped_data.reshape(1000, 96, 1366)
    shaped_labels = shaped_labels.reshape(1000, 10)

    x_train = shaped_data[0:train_size]
    x_test = shaped_data[train_size:]

    y_train = shaped_labels[0:train_size]
    y_test = shaped_labels[train_size:]


    print("x:", x_train.shape)
    print("y:", len(y_train))

    model.fit(x_train, y_train, epochs=N_EPOCH)#, steps_per_epoch=800)
    model.evaluate(x_test, y_test)