# -*- coding: utf-8 -*-
import os
import fnmatch
import shutil
import time

import numpy as np
import pandas as pd
from Bio import SeqIO
import re
from models import models
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt


def read_data(path):
    pssm_files = os.listdir(path)
    data = []
    labels = []

    for pssm_file in pssm_files:
        df = pd.read_csv(path + pssm_file, sep=',', header=None)
        df = np.asarray(df)
        label = int(pssm_file.split('_')[1])
        data.append(df)
        labels.append(label)

    data = np.asarray(data, dtype=object)
    labels = np.asarray(labels, dtype=int)
    return data, labels


def normalize(data):
    return data / 10


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # read data
    path = 'data/train/'
    data, labels = read_data(path)

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for train_index, val_index in skf.split(data, labels):

        print(train_index)
        # split data
        train_data = np.expand_dims(data[train_index], axis=-1)
        train_labels = labels[train_index]
        val_data = np.expand_dims(data[val_index], axis=-1)
        val_labels = labels[val_index]

        # normalize data
        train_data = np.asarray(normalize(train_data)).astype('float32')
        train_labels = np.asarray(train_labels).astype('float32')
        val_data = np.asarray(normalize(val_data)).astype('float32')
        val_labels = np.asarray(val_labels).astype('float32')

        print("number of train data: {}".format(len(train_data)))
        print("number of val data: {}".format(len(val_data)))

        # create model
        model = models()
        print(model.summary())

        weight = {0: 6, 1: 1}

        BATCH_SIZE = 64
        EPOCHS = 50

        # create callback
        callback = [

        ]

        # train model
        history = model.fit(
            train_data,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(val_data, val_labels),
            class_weight=weight,
            verbose=1,
            callbacks=callback
        )
        model.save('saved_models/' + get_model_name(i))
        plot_accuracy(history)
        plt.savefig('saved_plots/accuracy_{}.png'.format(i))
        plot_loss(history)
        plt.savefig('saved_plots/loss_{}.png'.format(i))
