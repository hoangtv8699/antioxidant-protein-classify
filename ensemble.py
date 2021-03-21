import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *

with open('data/independent_2_data_bert.npy', 'rb') as f:
    data_bert_val = np.load(f, allow_pickle=True)
with open('data/independent_2_labels_bert.npy', 'rb') as f:
    labels_bert_val = np.load(f, allow_pickle=True)

with open('data/independent_1_data_bert.npy', 'rb') as f:
    data_bert_val_1 = np.load(f, allow_pickle=True)
with open('data/independent_1_labels_bert.npy', 'rb') as f:
    labels_bert_val_1 = np.load(f, allow_pickle=True)

data_bert_val = data_bert_val[:, 0]
data_bert_val_1 = data_bert_val_1[:, 0]

clf1 = load('saved_models/RF_top1_1.joblib')
pre = clf1.predict_proba(data_bert_val)
pre = np.asarray(pre)

pre_1 = clf1.predict_proba(data_bert_val_1)
pre_1 = np.asarray(pre_1)

test_path = 'data/test/independent_2/'
test_path_1 = 'data/test/independent_1/'

data, labels = read_data(test_path)
data_1, labels_1 = read_data(test_path_1)
data = np.expand_dims(data, axis=-1).astype(np.float32)
data_1 = np.expand_dims(data_1, axis=-1).astype(np.float32)

path = "saved_models/103/"
model_paths = os.listdir(path)
model = []
for model_path in model_paths:
    model.append(tf.keras.models.load_model(path + model_path, compile=False))

vote = voting(model, data)
vote_1 = voting(model, data_1)
# ave = average(model, data)
# med = median(model, data)

ave = []
for i in range(len(pre)):
    tmp = [(pre[i][0] + vote[i][0]) / 2, (pre[i][1] + vote[i][1]) / 2]
    ave.append(tmp)

ave = np.asarray(ave)

print('SEN:' + str(sensitivity(labels, ave).numpy()))
print('SPE:' + str(specificity(labels, ave).numpy()))
print('ACC:' + str(acc(labels, ave).numpy()))
print('MCC:' + str(mcc(labels, ave).numpy()))
print('AUC:' + str(auc(labels, ave).numpy()))
ave_1 = []
for i in range(len(pre_1)):
    tmp = [pre_1[i][0] + vote_1[i][0], pre_1[i][1] + vote_1[i][1]]
    ave_1.append(tmp)

print('ACC:' + str(acc(labels_1, ave_1).numpy()))