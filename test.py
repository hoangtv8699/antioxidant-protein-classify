# -*- coding: utf-8 -*-
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from utils.helpers import *

if __name__ == '__main__':
    test_path = 'data/test/independent_2/'

    data, labels = read_data(test_path, padding="pad_sequence")
    data = normalize_data(data)
    data = np.expand_dims(data, axis=-1).astype(np.float32)

    model_paths = os.listdir("saved_models/3128/")
    model = []
    for model_path in model_paths:
        model.append(keras.models.load_model("saved_models/3128/" + model_path,
                                             custom_objects={"sensitivity": sensitivity,
                                                             "specificity": specificity,
                                                             "mcc": mcc,
                                                             }))

    i = 0
    for i in range(len(model)):
        pre = model[i].predict(data)
        print("model: " + str(i))
        print(sensitivity(labels, pre))
        print(specificity(labels, pre))
        print(acc(labels, pre))
        print(mcc(labels, pre))
        print(auc(labels, pre))
        i += 1

    vote = voting(model, data)
    ave = average(model, data)
    med = median(model, data)

    print("voting:")
    print("sen:" + str(sensitivity(labels, vote)))
    print("spe:" + str(specificity(labels, vote)))
    print("acc:" + str(acc(labels, vote)))
    print("mcc:" + str(mcc(labels, vote)))
    print("auc:" + str(auc(labels, vote)))

    print("ave:")
    print("sen:" + str(sensitivity(labels, ave)))
    print("spe:" + str(specificity(labels, ave)))
    print("acc:" + str(acc(labels, ave)))
    print("mcc:" + str(mcc(labels, ave)))
    print("auc:" + str(auc(labels, ave)))

    print("med:")
    print("sen:" + str(sensitivity(labels, med)))
    print("spe:" + str(specificity(labels, med)))
    print("acc:" + str(acc(labels, med)))
    print("mcc:" + str(mcc(labels, med)))
    print("auc:" + str(auc(labels, med)))
