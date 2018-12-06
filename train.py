import numpy as np
import tensorflow as tf

import sys

import matplotlib.pyplot as plt

import librosa as lb

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
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))

def buildLSTMModel(input_shape):
    # THIS MODEL FROM: https://github.com/anqiyu23/Introduction-to-Deep-Learning/blob/master/Final%20Project/code/lstm_genre_classifier_keras.py


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,
                                   input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=N_CATEGORIES, activation='softmax'))

    print("Compiling ...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model

def buildCNNModel(input_shape):
    # ours
    input_shape = input_shape + [1]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(4, 4),
            strides=(4, 4),  # X and Y to move the window by
            activation=tf.nn.sigmoid,
            input_shape=input_shape),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, (4, 4), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(N_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.categorical_crossentropy, # WHY DID WE DO THIS??
                  metrics=['accuracy'])


def train(model, model_type, x_train, x_test, y_train, y_test):
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

    print("Saving...")
    if (model_type == 2):
        model.save_weights('./LSTMweights')
        print("Saved to: ./LSTMweights")
    elif (model_type == 1):
        model.save_weights('./CNNweights')



    plt.plot(range(1, n_epoch+1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(range(1, n_epoch+1), history.loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return score

def getData():
    labels = np.load("labels.npy")
    data = np.load("data.npy")
    data = np.ndarray.astype(data, "float32")
    labels = np.ndarray.astype(labels, "int")
    # randomly shuffle data
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    x_train = data[0:train_size]
    x_test = data[train_size:]

    x_train = x_train.reshape([-1, 96, 1366])
    x_test = x_test.reshape([-1, 96, 1366])


    y_train = labels[0:train_size]
    y_test = labels[train_size:]


    print("x:", x_train.shape)
    print("y:", len(y_train))

    return x_train, x_test, y_train, y_test



def predict(model, path):
    classes = ["jazz", "blues", "reggae", "pop", "disco", "country", "metal", "hiphop", "rock", "classical"]

    score = model.evaluate(getData()[1], getData()[3])
    print('Pred loss:', score[0])
    print('Pred accuracy:', score[1])

    y, sr = lb.load(path, mono=True)
    spectogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=96, n_fft=2048, hop_length=256)
    spectogram = lb.power_to_db(spectogram, ref=np.max)

    predict_data = np.empty([1, 1366, 96])
    np.append(predict_data, spectogram)

    prediction = model.predict(predict_data.reshape(1,96, 1366))
    pred_class = prediction.argmax()

    print("I think that song is", classes[pred_class])
    print(prediction)


if __name__ == '__main__':
    model_type = 0

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
    elif len(sys.argv) == 1:
        try:
            model_type = int(input("1: CNN\n2: LSTM\n"))
            if model_type > 2 or model_type < 1:
                raise ValueError
        except ValueError:
            print("enter \'1\' or \'2\'")
            exit()
        mode = input("\'train\' (t) or \'predict\' (p) ?\n")
        if str.startswith(mode, "t"):
            mode = "train"
            try:
                batch_size = int(input("batch size: "))
                n_epoch = int(input("number of epochs: "))
            except ValueError:
                print("Please use ints for batch size and number of epochs")
                exit()

        elif str.startswith(mode, "p"):
            mode = "predict"
        else:
            print("enter t or p")
            quit()

    input_shape = (96, 1366)

    if (model_type == 2):
        model = buildLSTMModel(input_shape)
    elif (model_type == 1):
        model = buildCNNModel(input_shape)



    if(mode == "train"):
        x_train, x_test, y_train, y_test = getData()
        train(model, model_type, x_train, x_test, y_train, y_test)
    elif (mode == "predict"):
        if(model_type == 1):
            model.load_weights("./CNNweights")
        elif (model_type == 2):
            model.load_weights("./LSTMweights")

        path = input("path: ")

        predict(model,path)







