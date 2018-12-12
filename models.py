import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1338)
import random
random.seed(1339)

from keras import backend as K
K.set_image_dim_ordering('th')


# CNN_LEARN_RATE = 0.0015 # default: 0.0005
N_CATEGORIES = 10


def buildLSTMModel(input_shape):
    print("Building LSTM...")

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,
                                        input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=N_CATEGORIES, activation='softmax'))

    adam = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def buildCNNModel(input_shape, lr):
    print("Building CNN...")

    input_shape = input_shape + [1]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(4, 4),
                               strides=(4, 4),  # X and Y to move the window by
                               activation=tf.nn.relu,
                               input_shape=input_shape,
                               padding='SAME'),
        # tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Conv2D(64, (4, 4), activation=tf.nn.relu, padding='SAME'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='VALID'),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Conv2D(64, (4, 4), activation=tf.nn.relu, padding='SAME'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='VALID'),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Conv2D(64, (4, 4), activation=tf.nn.relu, padding='SAME'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='VALID'),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1000, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(N_CATEGORIES, activation='softmax')
    ])
    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam,
                       loss=tf.keras.losses.categorical_crossentropy,
                       metrics=['accuracy'])

    return model
