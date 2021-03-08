# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC

from utils.helpers import *
from utils.adasopt import AdasOptimizer


def models(maxlen=400):
    model = Sequential([
        Input(shape=(61, maxlen, 1)),

        ZeroPadding2D(1),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        ZeroPadding2D(1),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        ZeroPadding2D(1),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        ZeroPadding2D(1),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Flatten(),
        # Reshape((-1, 1)),
        #
        # LSTM(56),
        # Dropout(0.2),

        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.0001)
    # optimizer = AdasOptimizer(lr=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[sensitivity, specificity, mcc, 'accuracy']
    )
    return model


def train(n_splits, path_pssm, path_one_hot, path_blosum, batch_size, epochs, random_state, maxlen=400,
          save_path_model="saved_models/", save_path_his="saved_histories/"):
    # read data
    data_pssm, labels = read_data(path_pssm, padding="pad_sequence", maxlen=maxlen)
    data_one_hot, labels_one_hot = read_fasta(path_one_hot, maxlen=400, encode='onehot')
    data_blosum, labels_blosum = read_blosum(path_blosum, type='fasta')

    data_pssm = normalize_data(data_pssm)

    missing = check_missing(path_pssm)
    data_one_hot = np.delete(data_one_hot, missing, axis=0)
    data_blosum = np.delete(data_blosum, missing, axis=0)

    print("pssm shape: " + str(data_pssm.shape))
    print("one hot shape: " + str(data_one_hot.shape))

    data = np.append(data_pssm, data_blosum, axis=1)
    data = np.append(data, data_one_hot, axis=1)

    print("final shape: " + str(data.shape))

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    i = 0
    for train_index, val_index in skf.split(data, labels):
        # split data
        train_data = data[train_index]
        val_data = data[val_index]

        train_data = np.expand_dims(train_data, axis=-1).astype(np.float32)
        train_labels = labels[train_index]
        val_data = np.expand_dims(val_data, axis=-1).astype(np.float32)
        val_labels = labels[val_index]

        train_data, train_labels = balance_data(train_data, train_labels)

        train_posi = sum(train_labels)
        train_nega = len(train_labels) - train_posi
        val_posi = sum(val_labels)
        val_nega = len(val_labels) - val_posi

        print("number of train positive: {}".format(train_posi))
        print("number of train negative: {}".format(train_nega))
        print("number of val positive: {}".format(val_posi))
        print("number of val negative: {}".format(val_nega))

        print(train_labels.shape)

        # create model
        model = models(maxlen)
        print(model.summary())

        # create weight
        weight = {0: 1, 1: 6}

        # callback
        es = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode='min',
            restore_best_weights=True
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
            # class_weight=weight,
            callbacks=callbacks,
            shuffle=True,
            verbose=2
        )
        model.save(save_path_model + get_model_name(i))
        pickle.dump(history.history, open(save_path_his + "model_" + str(i), 'wb'), pickle.HIGHEST_PROTOCOL)
        # history = pickle.load(open(save_path_his + "model_" + str(i), 'wb'))
        i += 1


if __name__ == '__main__':
    path_one_hot = '../data/training.fasta'
    path_pssm = '../data/csv/'
    path_blosum = '../data/training.fasta'

    n_splits = 10
    # random_state = random.randint(0, 19999)
    random_state = 3568
    BATCH_SIZE = 16
    EPOCHS = 200
    print(random_state)
    save_path_model = "saved_models/" + str(random_state) + " one-hot blosum/"
    save_path_his = "saved_histories/" + str(random_state) + " one-hot blosum/"
    if not os.path.isdir(save_path_model):
        os.mkdir(save_path_model)
    if not os.path.isdir(save_path_his):
        os.mkdir(save_path_his)

    train(n_splits, path_pssm, path_one_hot, path_blosum, BATCH_SIZE, EPOCHS, random_state, maxlen=400,
          save_path_model=save_path_model, save_path_his=save_path_his)

