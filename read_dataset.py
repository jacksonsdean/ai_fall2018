import os
import sys
import numpy as np
import pandas as pd
import librosa as lb

from scipy import misc


# FROM https://github.com/meetshah1995/crnn-music-genre-classification/blob/master/src/read_data.py
labels_file = 'data/labels.csv'
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
labels = pd.read_csv(labels_file, header=0)


def get_labels(labels_dense=labels['label'], num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def get_melspectrograms(labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path']])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms


def get_melspectrograms_indexed(index, labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path'][index]])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms



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
        signal = signal[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref_power=1.0)

    if plot:
        melspect = melspect[np.newaxis, :]
        misc.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        print(melspect.shape)

    return melspect



batch_size    = 4
learning_rate = 0.003
n_epoch       = 50
n_samples     = 10                              # change to 1000 for entire dataset
cv_split      = 0.8
train_size    = int(n_samples * cv_split)
test_size     = n_samples - train_size

indices = np.arange(n_samples)
np.random.shuffle(indices)
train_indices = indices[0:train_size]
test_indices  = indices[train_size:]

labels = get_labels()

X_test = get_melspectrograms_indexed(test_indices)
y_train = labels[train_indices]
y_test = labels[test_indices]
