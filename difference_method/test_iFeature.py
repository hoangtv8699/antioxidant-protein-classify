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
    pssm_test_path = '../data/test/independent_2/'

    # read data
    data_pssm, labels = read_data(pssm_test_path, padding="pad_sequence", maxlen=400)

    data_AAC, labels_AAC = read_iFeature('../data/iFeature/independent_2/AAC.txt')
    data_APAAC, labels_APAAC = read_iFeature('../data/iFeature/independent_2/APAAC.txt')
    data_DDE, labels_APAAC = read_iFeature('../data/iFeature/independent_2/DDE.txt')
    data_DPC, labels_APAAC = read_iFeature('../data/iFeature/independent_2/DPC.txt')
    data_Geary, labels_APAAC = read_iFeature('../data/iFeature/independent_2/Geary.txt')
    data_Moran, labels_APAAC = read_iFeature('../data/iFeature/independent_2/Moran.txt')
    data_NMBroto, labels_APAAC = read_iFeature('../data/iFeature/independent_2/NMBroto.txt')
    data_SOCNumber, labels_APAAC = read_iFeature('../data/iFeature/independent_2/SOCNumber.txt')

    # missing = check_missing(path_pssm)
    # data_AAC = np.delete(data_AAC, missing, axis=0)
    new_shape = (data_AAC.shape[0], 20, int(data_AAC.shape[1] / 20))
    data_AAC = data_AAC.reshape(new_shape)

    new_shape = (data_APAAC.shape[0], 20, int(data_APAAC.shape[1] / 20))
    data_APAAC = data_APAAC.reshape(new_shape)

    new_shape = (data_DDE.shape[0], 20, int(data_DDE.shape[1] / 20))
    data_DDE = data_DDE.reshape(new_shape)

    new_shape = (data_DPC.shape[0], 20, int(data_DPC.shape[1] / 20))
    data_DPC = data_DPC.reshape(new_shape)

    new_shape = (data_Geary.shape[0], 20, int(data_Geary.shape[1] / 20))
    data_Geary = data_Geary.reshape(new_shape)

    new_shape = (data_Moran.shape[0], 20, int(data_Moran.shape[1] / 20))
    data_Moran = data_Moran.reshape(new_shape)

    new_shape = (data_NMBroto.shape[0], 20, int(data_NMBroto.shape[1] / 20))
    data_NMBroto = data_NMBroto.reshape(new_shape)

    new_shape = (data_SOCNumber.shape[0], 20, int(data_SOCNumber.shape[1] / 20))
    data_SOCNumber = data_SOCNumber.reshape(new_shape)
    #
    # data_APAAC = np.delete(data_APAAC, missing, axis=0)
    # new_shape = (data_APAAC.shape[0], 20, int(data_APAAC.shape[1] / 20))
    # data_APAAC = data_APAAC.reshape(new_shape)

    # print("pssm shape: " + str(data_pssm.shape))
    # print("iFeature shape: " + str(data_AAC.shape))
    data = np.append(data_AAC, data_APAAC, axis=2)
    data = np.append(data, data_DDE, axis=2)
    data = np.append(data, data_DPC, axis=2)
    data = np.append(data, data_Geary, axis=2)
    data = np.append(data, data_Moran, axis=2)
    data = np.append(data, data_NMBroto, axis=2)
    data = np.append(data, data_SOCNumber, axis=2)
    print("final shape: " + str(data.shape))

    data = np.expand_dims(data, axis=-1).astype(np.float32)

    common_path = "../saved_models/3518 APAAC/"
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
