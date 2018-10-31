import os
import sys
import math
import numpy as np
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import pickle as pk
import tensorflow as tf
import sklearn.metrics as sm

from scipy import misc


# FROM https://github.com/meetshah1995/crnn-music-genre-classification/blob/master/src/read_data.py
labels_file = 'labels.csv'
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
labels = pd.read_csv(labels_file, header=0)

#OURS:
def get_labels(labels_file):
    labels = pd.read_csv(labels_file, header=0)
    return labels


def get_melspectrograms(labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path']])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms


def get_melspectrograms_indexed(index, labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path'][index]])
    # spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms

#ours:
def get_train_data(labels):
    train_labels = []
    train_data = []
    print("Starting Data Process: ")
    for i in range(n_samples):
        path = labels['path'][i]
        label = int(labels['label'][i])
        train_labels.append(label)
        data = log_scale_melspectrogram(path)
        train_data.append(data)
        percent_done = round((i/n_samples)*100,2)
        print(str(percent_done) + "% done", end="\r")
    print()
    return np.array(train_labels), np.array(train_data)


Fs         = 12000
N_FFT      = 512
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 29.12

def log_scale_melspectrogram(path, plot=False):
    signal, sr = lb.load(path, sr=Fs)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        bottom = int(math.floor(n_sample-n_sample_fit)/2)
        top = int(math.floor(n_sample+n_sample_fit)/2)

        signal = signal[bottom:top]

    melspect = lb.amplitude_to_db(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref=1.0)

    if plot:
        melspect = melspect[np.newaxis, :]
        plt.imshow(melspect.reshape((melspect.shape[1], melspect.shape[2])))
        plt.show()
        print(melspect.shape)
    return melspect



n_epoch       = 10
n_samples     = 1000                              # 1000 for entire dataset
cv_split      = 0.8
train_size    = int(n_samples * cv_split)
test_size     = n_samples - train_size

if __name__ == '__main__':
    # Our net:
    labels = get_labels(labels_file)

    x_train, y_train = get_train_data(labels)

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


    model.fit(y_train, x_train, epochs=n_epoch)
    model.evaluate(y_test, y_test)