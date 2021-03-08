import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *

with open('data/train_data_bert.npy', 'rb') as f:
    data_bert = np.load(f, allow_pickle=True)
with open('data/train_labels_bert.npy', 'rb') as f:
    labels_bert = np.load(f, allow_pickle=True)

with open('data/independent_2_data_bert.npy', 'rb') as f:
    data_bert_val = np.load(f, allow_pickle=True)
with open('data/independent_2_labels_bert.npy', 'rb') as f:
    labels_bert_val = np.load(f, allow_pickle=True)

data_bert = data_bert[:, 0]
data_bert_val = data_bert_val[:, 0]

data_bert, labels_bert = balance_data(data_bert, labels_bert)

clf1 = load('saved_models/RFtop1.joblib')
pre = clf1.predict_proba(data_bert_val)
pre = np.asarray(pre)

test_path = 'data/test/independent_2/'

data, labels = read_data(test_path)
data = np.expand_dims(data, axis=-1).astype(np.float32)
path = "saved_models/3518/"
model_paths = os.listdir(path)
model = []
for model_path in model_paths:
    model.append(tf.keras.models.load_model(path + model_path,
                                            custom_objects={"sensitivity": sensitivity,
                                                            "specificity": specificity,
                                                            "mcc": mcc,
                                                            # 'AdasOptimizer': AdasOptimizer
                                                            }, compile=False))

vote = voting(model, data)
# ave = average(model, data)
# med = median(model, data)

ave = []
for i in range(len(pre)):
    tmp = [pre[i][0] + vote[i][0], pre[i][1] + vote[i][1]]
    ave.append(tmp)

print('SEN:' + str(sensitivity(labels, ave).numpy()))
print('SPE:' + str(specificity(labels, ave).numpy()))
print('ACC:' + str(acc(labels, ave).numpy()))
print('MCC:' + str(mcc(labels, ave).numpy()))
print('AUC:' + str(auc(labels, ave).numpy()))