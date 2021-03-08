import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *
from sklearn.utils import resample

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
posi = data_bert[labels_bert == 1]
nega = data_bert[labels_bert == 0]

posi = resample(posi, replace=True, n_samples=int(len(nega)), random_state=1)

data_bert = np.append(posi, nega, axis=0)
labels_posi = [1 for i in range(len(posi))]
labels_nega = [0 for i in range(len(nega))]
labels_bert = np.append(labels_posi, labels_nega)

print(len(posi))
print(len(nega))
print(labels_bert.shape)

clf = ensemble.RandomForestClassifier()

clf.fit(data_bert, labels_bert)
dump(clf, 'saved_models/RFtop1Resampling.joblib')

pre = clf.predict_proba(data_bert_val)
pre = np.asarray(pre)

print('SEN:' + str(sensitivity(labels_bert_val, pre).numpy()))
print('SPE:' + str(specificity(labels_bert_val, pre).numpy()))
print('ACC:' + str(acc(labels_bert_val, pre).numpy()))
print('MCC:' + str(mcc(labels_bert_val, pre).numpy()))
print('AUC:' + str(auc(labels_bert_val, pre).numpy()))
