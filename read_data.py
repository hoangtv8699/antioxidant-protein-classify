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
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau


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


def plot_loss(history, i):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('saved_plots/loss_{}.png'.format(i))
    plt.clf()


def plot_accuracy(history, i):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('saved_plots/accuracy_{}.png'.format(i))
    plt.clf()


def plot_sensitivity(history, i):
    plt.plot(history.history['sensitivity'])
    plt.plot(history.history['val_sensitivity'])
    plt.title('model sensitivity')
    plt.ylabel('sensitivity')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('saved_plots/sensitivity_{}.png'.format(i))
    plt.clf()


def plot_specificity(history, i):
    plt.plot(history.history['specificity'])
    plt.plot(history.history['val_specificity'])
    plt.title('model specificity')
    plt.ylabel('specificity')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('saved_plots/specificity_{}.png'.format(i))
    plt.clf()


def train(n_splits, path, batch_size, epochs):
    # read data
    data, labels = read_data(path)
    data = normalize(data)

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    i = 0
    for train_index, val_index in skf.split(data, labels):
        # split data
        train_data = np.expand_dims(data[train_index], axis=-1).astype(np.float32)
        train_labels = labels[train_index]
        val_data = np.expand_dims(data[val_index], axis=-1).astype(np.float32)
        val_labels = labels[val_index]

        print("number of train data: {}".format(len(train_data)))
        print("number of val data: {}".format(len(val_data)))

        # create model
        model = models()
        print(model.summary())

        # create weight
        weight = {0: 6, 1: 1}

        # callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                      patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=5,
                                      min_lr=0.0001)
        callbacks = [
            reduce_lr
        ]

        # train model
        history = model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            class_weight=weight,
            callbacks=callbacks
        )
        model.save('saved_models/' + get_model_name(i))
        plot_accuracy(history, i)
        plot_loss(history, i)
        plot_sensitivity(history, i)
        plot_specificity(history, i)
        break


if __name__ == '__main__':
    path = 'data/train/'
    n_splits = 5
    random_state = 1
    BATCH_SIZE = 64
    EPOCHS = 300
    train(n_splits, path, BATCH_SIZE, EPOCHS)
