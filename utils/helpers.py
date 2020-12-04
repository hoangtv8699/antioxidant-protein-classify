# -*- coding: utf-8 -*-
import os

import numpy as np
from Bio import SeqIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot

import pandas as pd
import tensorflow as tf
import random
import math
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(threshold=np.inf)


def normalize_data(datas):
    data_copy = np.copy(datas)
    for i in range(len(data_copy)):
        data = data_copy[i]
        for j in range(len(data)):
            row = data[j]
            sample_mean = sum(row) / (len(row) + K.epsilon())
            standard_deviation = math.sqrt(sum([pow(x - sample_mean, 2) for x in row]) / (len(row) + K.epsilon()))
            data[j] = [(x - sample_mean) / (standard_deviation + K.epsilon()) for x in row]
        data_copy[i] = data
    return data_copy


def normalize_common(data):
    sample_mean = 0
    for sequence in data:
        for row in sequence:
            sample_mean += sum(row)
    sample_mean /= (data.shape[0] * data.shape[1] * data.shape[2] + K.epsilon())

    standard_deviation = 0
    for sequence in data:
        for row in sequence:
            standard_deviation += sum([pow(x - sample_mean, 2) for x in row])
    standard_deviation = math.sqrt(standard_deviation / (data.shape[0] * data.shape[1] * data.shape[2] + K.epsilon()))

    data = (data - sample_mean) / (standard_deviation + K.epsilon())
    return data, sample_mean, standard_deviation


def normalize(data, sample_mean, standard_deviation):
    data = (data - sample_mean) / (standard_deviation + K.epsilon())
    return data


def balance_data(data, labels):
    posi = []
    nega = []
    balanced_data = []
    balanced_labels = []

    for i in range(len(data)):
        if labels[i] == 1:
            posi.append(data[i])
        else:
            nega.append(data[i])

    random.shuffle(posi)
    # j = 0
    # for i in range(len(nega)):
    #     if j < len(posi):
    #         balanced_data.append(posi[j])
    #         balanced_labels.append(1)
    #         j += 1
    #     else:
    #         j = 0
    #         random.shuffle(posi)
    #     balanced_data.append(nega[i])
    #     balanced_labels.append(0)

    j = 0
    for i in range(len(nega)):
        if i % 6 == 0:
            if j < len(posi):
                balanced_data.append(posi[j])
                balanced_labels.append(1)
                j += 1
        balanced_data.append(nega[i])
        balanced_labels.append(0)

    return np.asarray(balanced_data), np.asarray(balanced_labels)


def pad_same(sequence, maxlen=400):
    copy_sequence = np.copy(sequence)
    cur_len = len(sequence)
    if cur_len > maxlen:
        return sequence[:maxlen, :]
    step = int(maxlen / cur_len - 1)
    for i in range(step):
        sequence = np.append(sequence, copy_sequence, axis=0)
    append_len = maxlen - len(sequence)
    return np.append(sequence, copy_sequence[:append_len, :], axis=0)


def read_data(path, padding="pad_sequence"):
    pssm_files = os.listdir(path)
    data = []
    labels = []
    with np.printoptions(threshold=np.inf):
        for pssm_file in pssm_files:
            df = pd.read_csv(path + pssm_file, sep=',', header=None)
            df = np.asarray(df)
            if padding == "pad_sequence":
                df = sequence.pad_sequences(df.T, maxlen=400, padding='post', truncating='post').T
            elif padding == "same":
                df = pad_same(df, maxlen=400)
            label = int(pssm_file.split('_')[1])
            data.append(df.T)
            labels.append(label)

    data = np.asarray(data, dtype=object)
    labels = np.asarray(labels, dtype=int)
    return data, labels


def read_fasta(path, max_len=400, encode='token'):
    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(path), 'fasta')
    sequences = []
    labels = []
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id.split('|'), str(fasta.seq)
        sequences.append(sequence)
        labels.append(int(name[1]))

    tk = Tokenizer(num_words=21, char_level=True)
    # Fitting
    tk.fit_on_texts(sequences)
    print(tk.word_index)
    sequences, labels = pad_sequences(tk.texts_to_sequences(sequences), maxlen=max_len, padding='post',
                                      truncating='post'), np.asarray(labels)

    if encode == 'onehot':
        one_hot_sequences = []
        for sequence in sequences:
            b = np.zeros((400, 21))
            b[np.arange(400), sequence - 1] = 1
            one_hot_sequences.append(b.T)
        return np.asarray(one_hot_sequences), labels
    elif encode == 'token':
        return sequences, labels

    return None


def read_csv(path, max_len=400, encode='token'):
    df = pd.read_csv(path, skipinitialspace=True)

    sequences = []
    labels = []
    for index, row in df.iterrows():
        label = 1
        if row['SEQCLASS'] == 'non-antioxidant':
            label = 0
        sequences.append(row['SEQUENCE'])
        labels.append(label)

    tk = Tokenizer(num_words=21, char_level=True)
    # Fitting
    tk.fit_on_texts(sequences)
    sequences, labels = pad_sequences(tk.texts_to_sequences(sequences),
                                      maxlen=max_len, padding='post',
                                      truncating='post'), np.asarray(labels)

    if encode == 'onehot':
        one_hot_sequences = []
        for sequence in sequences:
            b = np.zeros((400, 21))
            b[np.arange(400), sequence - 1] = 1
            one_hot_sequences.append(b.T)
        return np.asarray(one_hot_sequences), labels
    elif encode == 'token':
        return sequences, labels

    return None


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


def sensitivity(y_true, y_pred):
    y_pred_bin = tf.math.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)
    sensitivity = TP / (TP + FN + K.epsilon())
    return sensitivity


def specificity(y_true, y_pred):
    y_pred_bin = tf.math.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)
    specificity = TN / (TN + FP + K.epsilon())
    return specificity


def mcc(y_true, y_pred):
    y_pred_bin = tf.math.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32).numpy()
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32).numpy()
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32).numpy()
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32).numpy()

    MCC = (TP * TN) - (FP * FN)
    MCC /= (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + K.epsilon())
    return MCC


def auc(y_true, y_pred):
    y_pred_bin = tf.math.argmax(y_pred, axis=-1)
    m = tf.keras.metrics.AUC()
    m.update_state(y_true, y_pred_bin)
    return m.result().numpy()


def voting(models, data):
    pre = []
    pre_bin = []
    for model in models:
        pre_tmp = model.predict(data)
        pre_bin.append(np.argmax(pre_tmp, axis=-1))

    for i in range(len(pre_bin[0])):
        tmp = [0, 0]
        for x in pre_bin:
            tmp[x[i]] += 1
        pre.append(tmp)
    return pre
