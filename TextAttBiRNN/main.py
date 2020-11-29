import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K

from TextAttBiRNN.text_att_birnn import TextAttBiRNN
from utils.helpers import *

if __name__ == '__main__':
    # read data
    path = 'training.fasta'
    data, labels = read_fasta(path, max_len=400)

    n_splits = 5
    random_state = 1
    max_features = 22
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    epochs = 30

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    i = 0
    for train_index, val_index in skf.split(data, labels):
        # split data
        train_data = data[train_index]
        train_labels = labels[train_index]
        val_data = data[val_index]
        val_labels = labels[val_index]

        train_data, train_labels = balance_data(train_data, train_labels)

        print("number of train data: {}".format(len(train_data)))
        print("number of val data: {}".format(len(val_data)))

        # create model
        print('Build model...')
        model = TextAttBiRNN(maxlen, max_features, embedding_dims, class_num=2, last_activation='softmax')
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy", sensitivity, specificity])

        # create weight
        weight = {0: 1, 1: 6}

        # callback
        es = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode='min'
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                      patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=5,
                                      min_lr=0.00001)
        callbacks = [
            reduce_lr,
            es
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
        # model.save('saved_models/' + get_model_name(i))
        plot_accuracy(history, i)
        plot_loss(history, i)
        plot_sensitivity(history, i)
        plot_specificity(history, i)
        break
