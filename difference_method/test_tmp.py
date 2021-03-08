# -*- coding: utf-8 -*-
import json

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from utils.helpers import *
from utils.iFeature_helper import *

if __name__ == '__main__':
    new_test_path = '../data/independent_2.fasta'
    # read data
    # data, labels = read_fasta(new_test_path, maxlen=400, encode='token')
    # data = encodes_amino_feature(data)[:, 1]

    data, labels = read_iFeature('../data/iFeature/independent_2/AAC.txt')

    data = np.asarray(data)
    labels = np.asarray(labels)
    print("final shape: " + str(data.shape))

    common_path = "../saved_models/3518 AAC dnn/"
    model_paths = os.listdir(common_path)
    model = []
    for model_path in model_paths:
        model.append(keras.models.load_model(common_path + model_path,
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
