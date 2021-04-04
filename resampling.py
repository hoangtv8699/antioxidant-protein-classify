import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


with open('data/train_data_bert.npy', 'rb') as f:
    data_bert = np.load(f, allow_pickle=True)
with open('data/train_labels_bert.npy', 'rb') as f:
    labels_bert = np.load(f, allow_pickle=True)

with open('data/independent_2_data_bert.npy', 'rb') as f:
    data_bert_val = np.load(f, allow_pickle=True)
with open('data/independent_2_labels_bert.npy', 'rb') as f:
    labels_bert_val = np.load(f, allow_pickle=True)

with open('data/independent_1_data_bert.npy', 'rb') as f:
    data_bert_val_1 = np.load(f, allow_pickle=True)
with open('data/independent_1_labels_bert.npy', 'rb') as f:
    labels_bert_val_1 = np.load(f, allow_pickle=True)

# with open('data/train_data_bert_600.npy', 'rb') as f:
#     data_bert = np.load(f, allow_pickle=True)
# with open('data/train_labels_bert_600.npy', 'rb') as f:
#     labels_bert = np.load(f, allow_pickle=True)
#
# with open('data/independent_2_data_bert_600.npy', 'rb') as f:
#     data_bert_val = np.load(f, allow_pickle=True)
# with open('data/independent_2_labels_bert_600.npy', 'rb') as f:
#     labels_bert_val = np.load(f, allow_pickle=True)
#
# with open('data/independent_1_data_bert_600.npy', 'rb') as f:
#     data_bert_val_1 = np.load(f, allow_pickle=True)
# with open('data/independent_1_labels_bert_600.npy', 'rb') as f:
#     labels_bert_val_1 = np.load(f, allow_pickle=True)

data_bert = data_bert[:, 0]
data_bert_val = data_bert_val[:, 0]
data_bert_val_1 = data_bert_val_1[:, 0]


# resampling
posi = data_bert[labels_bert == 1]
nega = data_bert[labels_bert == 0]
# up resample
posi = resample(posi, replace=True, n_samples=int(len(nega)), random_state=101)
# down resample
# nega = resample(nega, replace=True, n_samples=int(len(posi)), random_state=1010)
# concat data
print(len(posi))
print(len(nega))
labels_posi = [1 for i in range(len(posi))]
labels_nega = [0 for i in range(len(nega))]
data_bert = np.append(posi, nega, axis=0)
labels_bert = np.append(labels_posi, labels_nega)
data_bert, labels_bert = balance_data(data_bert, labels_bert, 31)

# # SMOTE
# sm = SMOTE(random_state=1)
# data_bert, labels_bert = sm.fit_resample(data_bert, labels_bert)

clf = ensemble.RandomForestClassifier(random_state=101)
clf.fit(data_bert, labels_bert)
dump(clf, 'saved_models/RF_up_resample_400.joblib')

clf = load('saved_models/RF_up_resample_400.joblib')

pre = clf.predict_proba(data_bert_val)
pre = np.asarray(pre)

print('independent test 2:')

sen = sensitivity(labels_bert_val, pre).numpy()
spe = specificity(labels_bert_val, pre).numpy()
accc = acc(labels_bert_val, pre).numpy()
mccc = mcc(labels_bert_val, pre).numpy()
aucc = auc(labels_bert_val, pre).numpy()
print('SEN:' + str(sen))
print('SPE:' + str(spe))
print('ACC:' + str(accc))
print('MCC:' + str(mccc))
print('AUC:' + str(aucc))

print('independent test 1:')
pre = clf.predict_proba(data_bert_val_1)
pre = np.asarray(pre)
accc2 = acc(labels_bert_val_1, pre).numpy()
print('ACC:' + str(accc2))

b = [math.floor(sen * 10000) / 10000, math.floor(spe * 10000) / 10000, math.floor(accc * 10000) / 10000,
     math.floor(mccc * 10000) / 10000, math.floor(aucc * 10000) / 10000, math.floor(accc2 * 10000) / 10000]
pd.DataFrame([b]).to_csv('test.csv')


