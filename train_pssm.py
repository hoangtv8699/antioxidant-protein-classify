# -*- coding: utf-8 -*-

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
from tensorflow.keras.models import Sequential

from utils.helpers import *


def models():
    model = Sequential([
        Input(shape=(20, 400, 1)),

        ZeroPadding2D(1),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        ZeroPadding2D(1),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        ZeroPadding2D(1),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        ZeroPadding2D(1),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Flatten(),
        # Reshape((-1, 1)),
        #
        # LSTM(64),
        # Dropout(0.2),

        Dense(16, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[sensitivity, specificity, 'accuracy']
    )
    return model


def train(n_splits, path, batch_size, epochs, random_state):
    # read data
    data, labels = read_data(path, padding="pad_sequence")
    data = normalize(data)

    # create 10-fold cross validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    i = 0
    for train_index, val_index in skf.split(data, labels):
        # split data
        train_data = np.expand_dims(data[train_index], axis=-1).astype(np.float32)
        train_labels = labels[train_index]
        val_data = np.expand_dims(data[val_index], axis=-1).astype(np.float32)
        val_labels = labels[val_index]

        train_data, train_labels = balance_data(train_data, train_labels)

        print("number of train data: {}".format(len(train_data)))
        print("number of val data: {}".format(len(val_data)))

        # create model
        model = models()
        print(model.summary())
        #
        # # create weight
        # weight = {0: 1, 1: 6}
        #
        # # callback
        # checkpoint = "saved_models"
        # es = EarlyStopping(
        #     monitor="val_loss",
        #     patience=50,
        #     mode='min'
        # )
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
        #                               patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=5,
        #                               min_lr=0.00001)
        # callbacks = [
        #     reduce_lr,
        #     es
        # ]
        #
        # # train model
        # history = model.fit(
        #     train_data,
        #     train_labels,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     validation_data=(val_data, val_labels),
        #     class_weight=weight,
        #     callbacks=callbacks
        # )
        # model.save('saved_models/' + get_model_name(i))
        # plot_accuracy(history, i)
        # plot_loss(history, i)
        # plot_sensitivity(history, i)
        # plot_specificity(history, i)
        break


if __name__ == '__main__':
    path = 'data/csv/'
    n_splits = 5
    random_state = 1
    BATCH_SIZE = 64
    EPOCHS = 300
    train(n_splits, path, BATCH_SIZE, EPOCHS, random_state)