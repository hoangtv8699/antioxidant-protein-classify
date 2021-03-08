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
    bert_test_path = 'data/bert_test_20/independent_2/'
    pssm_test_path = '../data/test/independent_2/'

    # read data
    data_pssm, labels = read_data(pssm_test_path, padding="pad_sequence", maxlen=400)
    # data_bert, labels_bert = read_bert(bert_test_path, padding="pad_sequence", maxlen=400)

    with open('../data/independent_2_data_bert.npy', 'rb') as f:
        data_bert = np.load(f, allow_pickle=True)
    with open('../data/independent_2_labels_bert.npy', 'rb') as f:
        labels_bert = np.load(f, allow_pickle=True)

    data_bert = data_bert[:, :10, :]

    print("pssm shape: " + str(data_pssm.shape))
    print("bert shape: " + str(data_bert.shape))
    # data_pssm = normalize_data(data_pssm)
    data = np.append(data_pssm, data_bert, axis=1)
    # data = np.append(data_bert, data_pssm, axis=1)

    # data = normalize_data(data)

    print("final shape: " + str(data.shape))

    data = np.expand_dims(data, axis=-1).astype(np.float32)

    model_paths = os.listdir("../saved_models/2538 bert top 10/")
    model = []
    for model_path in model_paths:
        model.append(keras.models.load_model("saved_models/2538 bert top 10/" + model_path,
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
