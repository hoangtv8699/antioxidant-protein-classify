import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
import numpy as np


def sensitivity(y_true, y_pred):
    """Compute the confusion matrix for a set of predictions.

        Parameters
        ----------
        y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
        y_true   : correct values for the set of samples used (must be binary: 0 or 1)

        Returns
        -------
        out : the sensitivity
        """

    y_pred_bin = tf.math.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin)

    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)
    sensitivity = TP / (TP + FN + K.epsilon())
    return sensitivity


def specificity(y_true, y_pred):
    """Compute the confusion matrix for a set of predictions.

    Parameters
    ----------
    y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
    y_true   : correct values for the set of samples used (must be binary: 0 or 1)

    Returns
    -------
    out : the specificity
    """

    y_pred_bin = tf.math.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin)

    # as Keras Tensors
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)
    specificity = TN / (TN + FP + K.epsilon())
    return specificity


def models():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(None, None, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Reshape((-1, 1)),

        LSTM(256),
        Dropout(0.2),

        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[sensitivity, specificity, 'accuracy',]
    )
    return model


def models_cnn():
    model = Sequential([
        Conv1D(256, 3, activation='relu', input_shape=(400, 20)),
        MaxPooling1D(3),
        Dropout(0.4),

        Conv1D(256, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.4),

        Reshape((-1, 1)),

        LSTM(64),
        Dropout(0.4),

        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[sensitivity, specificity, 'accuracy',]
    )
    return model
