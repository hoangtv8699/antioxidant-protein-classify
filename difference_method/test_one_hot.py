# -*- coding: utf-8 -*-
import json

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from utils.helpers import *
from utils.adasopt import *

if __name__ == '__main__':
    one_hot_test_path = '../data/independent_2.csv'
    pssm_test_path = '../data/test/independent_2/'
    blosum_test_path = '../data/independent_2.csv'

    # read data
    data_pssm, labels = read_data(pssm_test_path, padding="pad_sequence", maxlen=400)
    data_one_hot, labels_one_hot = read_csv(one_hot_test_path, maxlen=400, encode='onehot')
    data_blosum, labels_blosum = read_blosum(blosum_test_path, maxlen=400, type='csv')

    print("pssm shape: " + str(data_pssm.shape))
    print("bert shape: " + str(data_one_hot.shape))

    data_pssm = normalize_data(data_pssm)
    data = np.append(data_pssm, data_blosum, axis=1)
    data = np.append(data, data_one_hot, axis=1)

    print("final shape: " + str(data.shape))

    data = np.expand_dims(data, axis=-1).astype(np.float32)

    model_paths = os.listdir("saved_models/3568 one-hot blosum/")
    model = []
    for model_path in model_paths:
        model.append(keras.models.load_model("saved_models/3568 one-hot blosum/" + model_path,
                                             custom_objects={"sensitivity": sensitivity,
                                                             "specificity": specificity,
                                                             "mcc": mcc,
                                                             # 'AdasOptimizer': AdasOptimizer
                                                             }, compile=False))

    i = 0
    a = []
    b = []
    for i in range(len(model)):
        pre = model[i].predict(data)
        print("model: " + str(i))
        sen = sensitivity(labels, pre).numpy()
        spe = specificity(labels, pre).numpy()
        accc = acc(labels, pre).numpy()
        mccc = mcc(labels, pre).numpy()
        aucc = auc(labels, pre).numpy()
        b.append(math.floor(sen * 1000) / 1000)
        b.append(math.floor(spe * 1000) / 1000)
        b.append(math.floor(accc * 1000) / 1000)
        b.append(math.floor(mccc * 1000) / 1000)
        b.append(math.floor(aucc * 1000) / 1000)
        a.append(b)
        b = []
        i += 1

    vote = voting(model, data)
    ave = average(model, data)
    med = median(model, data)

    print("voting:")
    sen = sensitivity(labels, vote).numpy()
    spe = specificity(labels, vote).numpy()
    accc = acc(labels, vote).numpy()
    mccc = mcc(labels, vote).numpy()
    aucc = auc(labels, vote).numpy()
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)
    b = []

    print("ave:")
    sen = sensitivity(labels, ave).numpy()
    spe = specificity(labels, ave).numpy()
    accc = acc(labels, ave).numpy()
    mccc = mcc(labels, ave).numpy()
    aucc = auc(labels, ave).numpy()
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)
    b = []

    print("med:")
    sen = sensitivity(labels, med).numpy()
    spe = specificity(labels, med).numpy()
    accc = acc(labels, med).numpy()
    mccc = mcc(labels, med).numpy()
    aucc = auc(labels, med).numpy()
    b.append(math.floor(sen * 1000) / 1000)
    b.append(math.floor(spe * 1000) / 1000)
    b.append(math.floor(accc * 1000) / 1000)
    b.append(math.floor(mccc * 1000) / 1000)
    b.append(math.floor(aucc * 1000) / 1000)
    a.append(b)
    b = []

    pd.DataFrame(a).to_csv('../test.csv')
